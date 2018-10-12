import coremltools
import os

# model_dir = '../models/'
# caffe_model = (os.path.join(model_dir, 'squeezenet_v1.1.caffemodel'), os.path.join(model_dir, 'deploy.prototxt'))

# coreml_model = coremltools.converters.caffe.convert(caffe_model, image_input_names='data')
# coreml_model.save(os.path.join(model_dir, 'SqueezeNet_v1.1.mlmodel'))

import tvm
import nnvm
import nnvm.compiler

coreml_model = coremltools.models.MLModel(os.path.join(model_dir, 'SqueezeNet_v1.1.mlmodel')

sym, params = nnvm.frontend.from_coreml(coreml_model)
target = 'cuda'

shape_dict = {'data': (10, 3, 227, 227)}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

outfile = 'nnvm'

if target == 'llvm':
    outfile += '_llvm'

lib_file = outfile + '.so'
graph_file = outfile + '.json'
params_file = outfile + '.params'

lib.export_library(os.path.join(model_dir, lib_file))
with open(os.path.join(model_dir, graph_file), 'w') as fo:
    fo.write(graph.json())
with open(os.path.join(model_dir, params_file), 'wb') as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
