# simple_test

There are some comments in `train.py` that indicate lines that, when changed, make the error go away. If the error does not appear once, try running a few more times.

Prepare the environment:
```
conda create -n simple_test python=3.9
conda activate simple_test
pip install -r requirements.txt
```

Run:
```
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python train.py
```

Expected logs:
```
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 1 processes
----------------------------------------------------------------------------------------------------

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name  | Type   | Params
---------------------------------
0 | model | Linear | 10.2 K
---------------------------------
10.2 K    Trainable params
0         Non-trainable params
10.2 K    Total params
0.041     Total estimated model params size (MB)
/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:236: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 8 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(

Training: 0it [00:00, ?it/s]
Training:   0%|          | 0/2 [00:00<?, ?it/s]
Epoch 0:   0%|          | 0/2 [00:00<?, ?it/s] 
Epoch 0:  50%|█████     | 1/2 [00:01<00:01,  1.36s/it]
Epoch 0:  50%|█████     | 1/2 [00:01<00:01,  1.36s/it, loss=-1.12, v_num=1]
Epoch 0: 100%|██████████| 2/2 [00:01<00:00,  1.43it/s, loss=-1.12, v_num=1]
Epoch 0: 100%|██████████| 2/2 [00:01<00:00,  1.43it/s, loss=-3.72, v_num=1]
Epoch 0: 100%|██████████| 2/2 [00:01<00:00,  1.43it/s, loss=-3.72, v_num=1]
Epoch 0:   0%|          | 0/2 [00:00<?, ?it/s, loss=-3.72, v_num=1]        
Epoch 1:   0%|          | 0/2 [00:00<?, ?it/s, loss=-3.72, v_num=1]
Epoch 1:  50%|█████     | 1/2 [00:00<00:00,  3.75it/s, loss=-3.72, v_num=1]
Epoch 1:  50%|█████     | 1/2 [00:00<00:00,  3.72it/s, loss=-6.52, v_num=1]
Epoch 1: 100%|██████████| 2/2 [00:00<00:00,  6.87it/s, loss=-6.52, v_num=1]
Epoch 1: 100%|██████████| 2/2 [00:00<00:00,  6.84it/s, loss=-9.71, v_num=1]
Epoch 1: 100%|██████████| 2/2 [00:00<00:00,  6.77it/s, loss=-9.71, v_num=1]
Epoch 1:   0%|          | 0/2 [00:00<?, ?it/s, loss=-9.71, v_num=1]        
Epoch 2:   0%|          | 0/2 [00:00<?, ?it/s, loss=-9.71, v_num=1]terminate called after throwing an instance of 'c10::CUDAError'
  what():  CUDA error: initialization error
Exception raised from insert_events at ../c10/cuda/CUDACachingAllocator.cpp:1423 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x3e (0x7f2d18abe20e in /home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x23af2 (0x7f2d18b35af2 in /home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/lib/libc10_cuda.so)
frame #2: c10::cuda::CUDACachingAllocator::raw_delete(void*) + 0x257 (0x7f2d18b3a9a7 in /home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/lib/libc10_cuda.so)
frame #3: <unknown function> + 0x463338 (0x7f2d052ab338 in /home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/lib/libtorch_python.so)
frame #4: c10::TensorImpl::release_resources() + 0x175 (0x7f2d18aa57a5 in /home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #5: <unknown function> + 0x35f355 (0x7f2d051a7355 in /home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/lib/libtorch_python.so)
frame #6: <unknown function> + 0x678d38 (0x7f2d054c0d38 in /home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/lib/libtorch_python.so)
frame #7: THPVariable_subclass_dealloc(_object*) + 0x2b5 (0x7f2d054c10e5 in /home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/lib/libtorch_python.so)
frame #8: python() [0x4f0470]
frame #9: python() [0x4f0537]
frame #10: python() [0x5025a9]
frame #11: python() [0x4df64f]
frame #12: python() [0x4dd5d0]
frame #13: python() [0x590a9c]
frame #14: python() [0x553ab4]
<omitting python frames>
frame #16: python() [0x4e6a5a]
frame #19: python() [0x4e6a5a]
frame #22: python() [0x4e6a5a]
frame #25: python() [0x4e6a5a]
frame #28: python() [0x4f81a3]
frame #30: python() [0x4f81a3]
frame #32: python() [0x4e6a5a]
frame #33: python() [0x50547d]
frame #35: python() [0x4f81a3]
frame #37: python() [0x4f81a3]
frame #39: python() [0x5029a1]
frame #42: python() [0x4f81a3]
frame #44: python() [0x4f81a3]
frame #46: python() [0x4f81a3]
frame #48: python() [0x4e6a5a]
frame #50: python() [0x5029a1]
frame #53: python() [0x4f81a3]
frame #55: python() [0x4f81a3]
frame #56: python() [0x53045e]
frame #57: python() [0x5cb659]
frame #59: python() [0x4f8944]
frame #61: python() [0x4e6a5a]

Traceback (most recent call last):
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1163, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/queue.py", line 180, in get
    self.not_empty.wait(remaining)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/threading.py", line 316, in wait
    gotit = waiter.acquire(True, timeout)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/utils/data/_utils/signal_handling.py", line 66, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 4971) is killed by signal: Aborted. 

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ankit/devel/simple_test/train.py", line 72, in <module>
    main()
  File "/home/ankit/devel/simple_test/train.py", line 68, in main
    trainer.fit(module, datamodule=datamodule)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 696, in fit
    self._call_and_handle_interrupt(
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 648, in _call_and_handle_interrupt
    return self.strategy.launcher.launch(trainer_fn, *args, trainer=self, **kwargs)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/strategies/launchers/subprocess_script.py", line 93, in launch
    return function(*args, **kwargs)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 735, in _fit_impl
    results = self._run(model, ckpt_path=self.ckpt_path)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1166, in _run
    results = self._run_stage()
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1252, in _run_stage
    return self._run_train()
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1283, in _run_train
    self.fit_loop.run()
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/loops/loop.py", line 200, in run
    self.advance(*args, **kwargs)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py", line 271, in advance
    self._outputs = self.epoch_loop.run(self._data_fetcher)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/loops/loop.py", line 200, in run
    self.advance(*args, **kwargs)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/loops/epoch/training_epoch_loop.py", line 174, in advance
    batch = next(data_fetcher)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/utilities/fetching.py", line 184, in __next__
    return self.fetching_function()
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/utilities/fetching.py", line 263, in fetching_function
    self._fetch_next_batch(self.dataloader_iter)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/utilities/fetching.py", line 277, in _fetch_next_batch
    batch = next(iterator)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/supporters.py", line 557, in __next__
    return self.request_next_batch(self.loader_iters)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/supporters.py", line 569, in request_next_batch
    return apply_to_collection(loader_iters, Iterator, next)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/utilities/apply_func.py", line 99, in apply_to_collection
    return function(data, *args, **kwargs)
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 681, in __next__
    data = self._next_data()
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1359, in _next_data
    idx, data = self._get_data()
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1315, in _get_data
    success, data = self._try_get_data()
  File "/home/ankit/.miniconda3/envs/simple_test/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1176, in _try_get_data
    raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
RuntimeError: DataLoader worker (pid(s) 4971) exited unexpectedly
```
