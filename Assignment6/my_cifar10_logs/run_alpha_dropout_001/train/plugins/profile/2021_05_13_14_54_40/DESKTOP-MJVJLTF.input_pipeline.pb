	'?Wj2@'?Wj2@!'?Wj2@	???`-?????`-??!???`-??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$'?Wj2@o???T???AOjM2@YM?J???*	??????Y@2U
Iterator::Model::ParallelMapV2P?s???!??؉??@@)P?s???1??؉??@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???QI??!    ?;@)?b?=y??1?;??6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?0?*???!;?;?S3@)y?&1???1?N??N?*@:Preprocessing2F
Iterator::ModelJ+???!??N?ĎG@)???߾??1??؉?X*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)\???(??!;?;qJ@) ?o_?y?1?;?;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?HP?x?!b'vb'v@)?HP?x?1b'vb'v@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorU???N@s?!;?;?@)U???N@s?1;?;?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???`-??IE_~???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	o???T???o???T???!o???T???      ??!       "      ??!       *      ??!       2	OjM2@OjM2@!OjM2@:      ??!       B      ??!       J	M?J???M?J???!M?J???R      ??!       Z	M?J???M?J???!M?J???b      ??!       JCPU_ONLYY???`-??b qE_~???X@