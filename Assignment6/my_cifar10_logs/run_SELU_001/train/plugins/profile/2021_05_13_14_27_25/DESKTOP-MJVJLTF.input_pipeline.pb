	?f??j{5@?f??j{5@!?f??j{5@	Ũ/{????Ũ/{????!Ũ/{????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?f??j{5@؁sF????A@a???4@Y4??@????*	43333b@2U
Iterator::Model::ParallelMapV2n????!??q?4;@)n????1??q?4;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat	??g????!ƅ????:@)???{????1*?????6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???x?&??!?Y1l?7@)?{??Pk??1?(;?{?1@:Preprocessing2F
Iterator::Model?q??????!?T?FѦE@)???????1??R?0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?L?J???!??.YL@)Έ?????1h?r??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice? ?	??!k?4w?_@)? ?	??1k?4w?_@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor ?o_?y?!oζ?|@) ?o_?y?1oζ?|@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Ũ/{????IWЄ.4?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	؁sF????؁sF????!؁sF????      ??!       "      ??!       *      ??!       2	@a???4@@a???4@!@a???4@:      ??!       B      ??!       J	4??@????4??@????!4??@????R      ??!       Z	4??@????4??@????!4??@????b      ??!       JCPU_ONLYYŨ/{????b qWЄ.4?X@