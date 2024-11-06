# mmseg/core/hooks/progressive_expansion_hook.py

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmseg.registry import HOOKS
import copy
import torch
import torch.distributed as dist

@HOOKS.register_module()
class ProgressiveExpansionHook(Hook):
    def __init__(self, max_steps=14):
        self.max_steps = max_steps
        self.current_step = 0
        self.best_archs = []
        self.init_arch_params()
    
    def init_arch_params(self):
        # Initial architecture parameters
        self.depth = [1, 1, 1, 1, 1]
        self.width = [4, 8, 16, 32, 32]
        self.resolution = [1 / 2, 0, 0]
    
        # Delta values for expansion
        self.delta_depth = [
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1]
        ]
        self.delta_width = [
            [4, 8, 16, 32, 32],
            [0, 8, 16, 32, 32],
            [0, 0, 16, 32, 32],
            [0, 0, 0, 32, 32]
        ]
        self.delta_resolution = [
            [1/8, 0, 0],
            [0, 1/8, 0],
            [0, 0, 1/8]
        ]
    
    def before_run(self, runner: Runner):
        # Initialize the model with the smallest architecture
        self.current_arch = {
            'depth': self.depth,
            'width': self.width,
            'resolution': self.resolution
        }
        # Update the model architecture
        self.update_model_architecture(runner)
        # Evaluate initial mIoU and latency
        self.current_miou = self.evaluate_miou(runner)
        self.current_latency = self.measure_latency(runner)
        self.best_archs.append(copy.deepcopy(self.current_arch))
        self.current_step = 0
        if runner.world_rank == 0:
            print(f'Initial Architecture: {self.current_arch}')
            print(f'Initial mIoU: {self.current_miou}, Latency: {self.current_latency}')
    
    def before_train_epoch(self, runner: Runner):
        if self.current_step >= self.max_steps:
            return  # Stop expanding after max_steps
    
        if runner.world_rank == 0:
            print(f'\nStep {self.current_step + 1}/{self.max_steps}')
        # Expand the architecture
        expanded_archs = self.expand_all()
        # Evaluate expanded architectures
        best_arch, best_miou, best_latency = self.select_best_architecture(expanded_archs, runner)
        # Update the current architecture
        self.current_arch = best_arch
        self.current_miou = best_miou
        self.current_latency = best_latency
        self.best_archs.append(copy.deepcopy(self.current_arch))
        self.current_step += 1
        # Update the model with the new best architecture
        self.update_model_architecture(runner)
        if runner.world_rank == 0:
            print(f'Updated Architecture: {self.current_arch}')
            print(f'Updated mIoU: {self.current_miou}, Latency: {self.current_latency}')
    
    def update_model_architecture(self, runner: Runner):
        # Update the backbone and decode head
        if runner.world_rank == 0:
            backbone = runner.model.backbone
            backbone.update_architecture(
                depth=self.current_arch['depth'],
                width=self.current_arch['width'],
                resolution=self.current_arch['resolution']
            )
            backbone.init_weights()
    
            # Update the decode head
            decode_head = runner.model.decode_head
            new_in_channels = backbone.get_out_channels()
            decode_head.update_channels(new_in_channels)
            decode_head.init_weights()
        # Synchronize the updated model across GPUs
        self.sync_model_parameters(runner)
    
    def sync_model_parameters(self, runner: Runner):
        # Synchronize model parameters across GPUs
        for param in runner.model.parameters():
            dist.broadcast(param.data, src=0)
        # Additionally, synchronize buffers (e.g., BatchNorm running stats)
        for buffer in runner.model.buffers():
            dist.broadcast(buffer.data, src=0)
    
    def expand_all(self):
        # Generate all possible expansions
        expanded_archs = []
        # Expand depth
        for op in self.delta_depth:
            new_depth = [i + j for i, j in zip(self.current_arch['depth'], op)]
            if all(d > 0 for d in new_depth):
                new_arch = copy.deepcopy(self.current_arch)
                new_arch['depth'] = new_depth
                expanded_archs.append(new_arch)
        # Expand width
        for op in self.delta_width:
            new_width = [i + j for i, j in zip(self.current_arch['width'], op)]
            if all(w > 0 for w in new_width):
                new_arch = copy.deepcopy(self.current_arch)
                new_arch['width'] = new_width
                expanded_archs.append(new_arch)
        # Expand resolution
        for op in self.delta_resolution:
            new_resolution = [i + j for i, j in zip(self.current_arch['resolution'], op)]
            if all(r > 0 for r in new_resolution):
                new_arch = copy.deepcopy(self.current_arch)
                new_arch['resolution'] = new_resolution
                expanded_archs.append(new_arch)
        return expanded_archs
    
    def select_best_architecture(self, expanded_archs, runner: Runner):
        best_slope = float('-inf')
        best_arch = None
        best_miou = None
        best_latency = None
        for arch in expanded_archs:
            # Update model architecture
            if runner.world_rank == 0:
                runner.model.backbone.update_architecture(
                    depth=arch.get('depth', self.current_arch['depth']),
                    width=arch.get('width', self.current_arch['width']),
                    resolution=arch.get('resolution', self.current_arch['resolution'])
                )
                runner.model.backbone.init_weights()
                # Update decode head
                new_in_channels = runner.model.backbone.get_out_channels()
                runner.model.decode_head.update_channels(new_in_channels)
                runner.model.decode_head.init_weights()
            # Synchronize the updated model across GPUs
            self.sync_model_parameters(runner)
            # Measure latency and mIoU
            miou = self.evaluate_miou(runner)
            latency = self.measure_latency(runner)
            # Calculate the slope (delta mIoU / delta latency)
            delta_miou = miou - self.current_miou
            delta_latency = latency - self.current_latency
            slope = delta_miou / delta_latency if delta_latency > 0 else float('-inf')
            if runner.world_rank == 0:
                print(f'Evaluated Arch: {arch}, mIoU: {miou}, Latency: {latency}, Slope: {slope}')
            # Select the architecture with the best slope
            if slope > best_slope:
                best_slope = slope
                best_arch = arch
                best_miou = miou
                best_latency = latency
        return best_arch, best_miou, best_latency
    
    def evaluate_miou(self, runner: Runner):
        # Evaluate mIoU using runner.test()
        # Backup the original evaluator and dataloader
        original_val_evaluator = runner.val_evaluator
        original_val_loop = runner.val_loop
        original_val_dataloader = runner.val_dataloader

        # Temporarily set the evaluator and dataloader to validation ones
        runner.val_evaluator = runner.test_evaluator
        runner.val_loop = runner.test_loop
        runner.val_dataloader = runner.test_dataloader

        # Run evaluation
        eval_results = runner.test()  # This runs the test loop

        # Restore the original evaluator and dataloader
        runner.val_evaluator = original_val_evaluator
        runner.val_loop = original_val_loop
        runner.val_dataloader = original_val_dataloader

        # Get mIoU from evaluation results
        miou = eval_results['mIoU']
        # Broadcast mIoU to all ranks
        miou_tensor = torch.tensor(miou, device=next(runner.model.parameters()).device)
        dist.broadcast(miou_tensor, src=0)
        return miou_tensor.item()
    
    def measure_latency(self, runner: Runner):
        # Measure latency of the model
        model = runner.model
        device = next(model.parameters()).device
        model.eval()
        # Create dummy input based on current resolution
        base_resolution = (1024, 2048)
        scale_factors = [r if r > 0 else 1 for r in self.current_arch['resolution']]
        input_height = int(base_resolution[0] * scale_factors[0])
        input_width = int(base_resolution[1] * scale_factors[0])
        dummy_input = torch.randn(1, 3, input_height, input_width).to(device)
        # Ensure only rank 0 measures latency
        if runner.world_rank == 0:
            with torch.no_grad():
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                # Warm-up
                for _ in range(10):
                    _ = model(dummy_input)
                # Measure latency
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                latency = starter.elapsed_time(ender)  # milliseconds
            latency /= 1000.0  # Convert to seconds
        else:
            latency = 0.0
        # Broadcast latency to all ranks
        latency_tensor = torch.tensor(latency, device=device)
        dist.broadcast(latency_tensor, src=0)
        return latency_tensor.item()
