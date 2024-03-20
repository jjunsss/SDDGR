# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

def to_cuda(samples, targets, device, pseudo_labeling=False):
    samples = samples.to(device, non_blocking=True)
    if  pseudo_labeling :
        return samples
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True, Mosaic=False, pseudo_labeling=False):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        self.Mosaic = Mosaic
        self.data_gen = None
        self.pseudo_labeling = pseudo_labeling
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self, new = False):
        try:
            if self.Mosaic == True:
                if self.data_gen is None:
                    # in AugReplay. next_Current_target, next_Current_samples are next_replay_samples, next_replay_targets
                    # in mosaic. next_Current_target, next_Current_samples are mosaic_targets, mosaic_samples
                    self.next_samples, self.next_targets, self.next_Current_samples, self.next_Current_target = next(self.loader)
                    temp = [[self.next_samples, self.next_targets ], [self.next_Current_samples, self.next_Current_target]]
                        
                    self.data_gen = self._split_gpu_preload(temp)
                    self.next_samples, self.next_targets = next(self.data_gen)
                elif self.data_gen is not None:
                    self.next_samples, self.next_targets = next(self.data_gen, (None, None))
                    
                    if self.next_samples is None or new == True:
                        # in AugReplay. next_Current_target, next_Current_samples are next_replay_samples, next_replay_targets
                        self.next_samples, self.next_targets, self.next_Current_samples, self.next_Current_target = next(self.loader)
                        temp = [[self.next_samples, self.next_targets], [self.next_Current_samples, self.next_Current_target]]
                        self.data_gen = self._split_gpu_preload(temp)
                        self.next_samples, self.next_targets = next(self.data_gen, (None, None))
            else:
                self.next_samples, self.next_targets = next(self.loader)
                
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            self.next_Current_samples = None
            self.next_Current_targets = None
            return
        
        except KeyError:
            self.next_samples = None
            self.next_targets = None
            self.next_Current_samples = None
            self.next_Current_targets = None
            return
        
        with torch.cuda.stream(self.stream):
            if self.pseudo_labeling is True:
                self.next_samples = to_cuda(self.next_samples, self.next_targets, self.device, self.pseudo_labeling )
            else:    
                self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
    
    def _split_gpu_preload(self, temp):
        for samples, targets  in temp:
            yield samples, targets
                
    def next(self, new = False):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if self.pseudo_labeling :
                self.preload(new) #* change to next samples
                return samples, targets #* NestedTensor, targets = "image_Id"
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload(new)
        else:
            #TODO: if I do not use prefetch, I don't have to fix here.
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
                
        return samples, targets
