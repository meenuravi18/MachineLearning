	V-*I@V-*I@!V-*I@	??c?pN7@??c?pN7@!??c?pN7@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$V-*I@	?^)K@A+??	?A@Y?(\??u'@*	gfff???@2F
Iterator::Model??+eb'@!79|???X@)	?^)K'@1??>XO?X@:Preprocessing2U
Iterator::Model::ParallelMapV2Ǻ?????!?t9=?[??)Ǻ?????1?t9=?[??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?ܵ?|У?!P},g
??)??6???1??e?w??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX9??v???!D???????)F%u???1M ?#???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice???????![]??::??)???????1[]??::??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipH?}8g??!?d??????){?G?z??1h??p???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?r?!???V??)HP?s?r?1???V??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 23.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s7.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??c?pN7@I?'?c,S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
		?^)K@	?^)K@!	?^)K@      ??!       "      ??!       *      ??!       2	+??	?A@+??	?A@!+??	?A@:      ??!       B      ??!       J	?(\??u'@?(\??u'@!?(\??u'@R      ??!       Z	?(\??u'@?(\??u'@!?(\??u'@b      ??!       JCPU_ONLYY??c?pN7@b q?'?c,S@