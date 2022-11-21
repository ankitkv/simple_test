# simple_test

Run like:
```
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python train.py datamodule.data_path=<absolute path to data dir where MNIST will be downloaded>
```

Expected logs:
```
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
[2022-11-21 17:35:29,151][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 0
[2022-11-21 17:35:29,152][torch.distributed.distributed_c10d][INFO] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 1 nodes.
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 1 processes
----------------------------------------------------------------------------------------------------

Missing logger folder: /home/mila/v/vanianki/devel/simple_test/outputs/2022-11-21/17-35-28/lightning_logs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name  | Type              | Params
--------------------------------------------
0 | model | VisionTransformer | 21.5 M
1 | head  | Linear            | 49.3 K
--------------------------------------------
21.5 M    Trainable params
0         Non-trainable params
21.5 M    Total params
86.073    Total estimated model params size (MB)
Sanity Checking: 0it [00:00, ?it/s]/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:236: PossibleUserWarning: The dataloader, val_dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 64 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:236: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 64 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
Epoch 0:  25%|████████████████████████████████████▊                                                                                                              | 1/4 [00:11<00:35, 11.88s/it, loss=-646, v_num=0][2022-11-21 17:35:48,319][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
Epoch 0:  50%|████████████████terminate called after throwing an instance of 'c10::CUDAError'                                                               | 2/4 [00:12<00:12,  6.07s/it, loss=-3.17e+03, v_num=0]
  what():  CUDA error: initialization error
Exception raised from insert_events at ../c10/cuda/CUDACachingAllocator.cpp:1423 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x3e (0x7f7da89d920e in /home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x23af2 (0x7f7dd108eaf2 in /home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/lib/libc10_cuda.so)
frame #2: c10::cuda::CUDACachingAllocator::raw_delete(void*) + 0x257 (0x7f7dd10939a7 in /home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/lib/libc10_cuda.so)
frame #3: <unknown function> + 0x463338 (0x7f7dfa601338 in /home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/lib/libtorch_python.so)
frame #4: c10::TensorImpl::release_resources() + 0x175 (0x7f7da89c07a5 in /home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #5: <unknown function> + 0x35f355 (0x7f7dfa4fd355 in /home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/lib/libtorch_python.so)
frame #6: <unknown function> + 0x678d38 (0x7f7dfa816d38 in /home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/lib/libtorch_python.so)
frame #7: THPVariable_subclass_dealloc(_object*) + 0x2b5 (0x7f7dfa8170e5 in /home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/lib/libtorch_python.so)
<omitting python frames>

Error executing job with overrides: ['datamodule.data_path=/home/mila/v/vanianki/devel/simple_test/data']
Traceback (most recent call last):
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1163, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/queue.py", line 180, in get
    self.not_empty.wait(remaining)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/threading.py", line 316, in wait
    gotit = waiter.acquire(True, timeout)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/utils/data/_utils/signal_handling.py", line 66, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 40427) is killed by signal: Aborted. 

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/mila/v/vanianki/devel/simple_test/train.py", line 89, in <module>
    main()
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/hydra/main.py", line 48, in decorated_main
    _run_hydra(
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/hydra/_internal/utils.py", line 377, in _run_hydra
    run_and_report(
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/hydra/_internal/utils.py", line 214, in run_and_report
    raise ex
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/hydra/_internal/utils.py", line 211, in run_and_report
    return func()
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/hydra/_internal/utils.py", line 378, in <lambda>
    lambda: hydra.run(
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/hydra/_internal/hydra.py", line 111, in run
    _ = ret.return_value
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/hydra/core/utils.py", line 233, in return_value
    raise self._return_value
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/hydra/core/utils.py", line 160, in run_job
    ret.return_value = task_function(task_cfg)
  File "/home/mila/v/vanianki/devel/simple_test/train.py", line 82, in main
    trainer.fit(
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 696, in fit
    self._call_and_handle_interrupt(
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 648, in _call_and_handle_interrupt
    return self.strategy.launcher.launch(trainer_fn, *args, trainer=self, **kwargs)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/strategies/launchers/subprocess_script.py", line 93, in launch
    return function(*args, **kwargs)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 735, in _fit_impl
    results = self._run(model, ckpt_path=self.ckpt_path)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1166, in _run
    results = self._run_stage()
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1252, in _run_stage
    return self._run_train()
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1283, in _run_train
    self.fit_loop.run()
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/loops/loop.py", line 200, in run
    self.advance(*args, **kwargs)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py", line 271, in advance
    self._outputs = self.epoch_loop.run(self._data_fetcher)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/loops/loop.py", line 201, in run
    self.on_advance_end()
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/loops/epoch/training_epoch_loop.py", line 241, in on_advance_end
    self._run_validation()
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/loops/epoch/training_epoch_loop.py", line 299, in _run_validation
    self.val_loop.run()
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/loops/loop.py", line 200, in run
    self.advance(*args, **kwargs)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/loops/dataloader/evaluation_loop.py", line 155, in advance
    dl_outputs = self.epoch_loop.run(self._data_fetcher, dl_max_batches, kwargs)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/loops/loop.py", line 200, in run
    self.advance(*args, **kwargs)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/loops/epoch/evaluation_epoch_loop.py", line 127, in advance
    batch = next(data_fetcher)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/utilities/fetching.py", line 184, in __next__
    return self.fetching_function()
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/utilities/fetching.py", line 263, in fetching_function
    self._fetch_next_batch(self.dataloader_iter)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/pytorch_lightning/utilities/fetching.py", line 277, in _fetch_next_batch
    batch = next(iterator)
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 681, in __next__
    data = self._next_data()
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1359, in _next_data
    idx, data = self._get_data()
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1315, in _get_data
    success, data = self._try_get_data()
  File "/home/mila/v/vanianki/.conda/envs/simple_test/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1176, in _try_get_data
    raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
RuntimeError: DataLoader worker (pid(s) 40427) exited unexpectedly
```
