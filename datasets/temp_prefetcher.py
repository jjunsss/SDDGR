import torch

class DataPrefetcher():
    def __init__(self, loader, device, prefetch=True, Mosaic=False, Continual_Batch=2):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        self.Mosaic = Mosaic
        self.data_gen = None
        self.Continual_Batch = Continual_Batch
        self.stream = torch.cuda.Stream() if prefetch else None

        if prefetch:
            self.preload()

    def preload(self):
        try:
            if self.Mosaic == True and (self.data_gen is None or not self.next_sample()):
                data = next(self.loader)
                temp = self._generate_temp_data(data)
                self.data_gen = self._split_gpu_preload(temp)
            elif not self.Mosaic:
                self._set_next_samples(*next(self.loader))
        except StopIteration:
            self._reset_next_samples()
        except KeyError:
            self._reset_next_samples()
        
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = self.to_cuda(self.next_samples, self.next_targets, self.device)

    def _generate_temp_data(self, data):
        if self.Continual_Batch == 3:
            self._set_next_samples(*data)
            temp = [[self.next_samples, self.next_targets, self.next_origin_samples, self.next_origin_targets], 
                    [self.next_Current_samples, self.next_Current_target, None, None], # for augreplay & mosaic batch
                    [self.next_Diff_samples, self.next_Diff_targets, None, None]]  # formosaic batch
        else: # CB = 2
            self._set_next_samples(*data[:6])
            temp = [[self.next_samples, self.next_targets, self.next_origin_samples, self.next_origin_targets], 
                    [self.next_Current_samples, self.next_Current_target, None, None]]
        return temp
    
    def _set_next_samples(self, *data):
        (self.next_samples, self.next_targets, self.next_origin_samples, 
         self.next_origin_targets, self.next_Current_samples, 
         self.next_Current_target, self.next_Diff_samples, self.next_Diff_targets) = data + (None,) * (8 - len(data))

    def _reset_next_samples(self):
        self._set_next_samples(None, None, None, None, None, None, None, None)

    def next_sample(self):
        return self.next_samples or self.next_targets or self.next_origin_samples or self.next_origin_targets

    def _split_gpu_preload(self, temp):
        for samples, targets, origin_samples, origin_targets in temp:
            yield (samples, targets, origin_samples, origin_targets) if origin_samples is not None else (samples, targets, None, None)
                
    def next(self, new = False):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            origin_samples = self.next_origin_samples
            origin_targets = self.next_origin_targets

            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for v in t.values():
                        v.record_stream(torch.cuda.current_stream())
            self.preload(new)
        else:
            try:
                samples, targets, origin_samples, origin_targets, current_samples, current_targets, Diff_samples, Diff_targets = self.to_cuda(*next(self.loader), self.device)
            except StopIteration:
                samples = None
                targets = None
                
        return samples, targets, origin_samples, origin_targets

    @staticmethod
    def to_cuda(*args, device):
        return tuple(arg.to(device) if arg is not None else None for arg in args)
