	q=
ף?9@q=
ף?9@!q=
ף?9@	p?}g???p?}g???!p?}g???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$q=
ף?9@A?c?]K??AL?
F%?8@Yk?w??#??*	??????e@2U
Iterator::Model::ParallelMapV2=?U?????!?????j;@)=?U?????1?????j;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??6???!t????>@)??ZӼ???1~?~L57@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat46<???!%?|]??4@)??????1??à??1@:Preprocessing2F
Iterator::Model?n??ʱ?!W???K?C@)䃞ͪϕ?1??Z?i;(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???z6??!?G?;N@)X9??v???1[G%?8?!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceS?!?uq??!G?;}@)S?!?uq??1G?;}@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?+e?Xw?!P???E?	@)?+e?Xw?1P???E?	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9p?}g???I?-??j?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	A?c?]K??A?c?]K??!A?c?]K??      ??!       "      ??!       *      ??!       2	L?
F%?8@L?
F%?8@!L?
F%?8@:      ??!       B      ??!       J	k?w??#??k?w??#??!k?w??#??R      ??!       Z	k?w??#??k?w??#??!k?w??#??b      ??!       JCPU_ONLYYp?}g???b q?-??j?X@