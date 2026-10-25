from pathlib import Path
import numpy as np
import pytest
from NeedleFinderLib import NeuralNeedleNet as nn

pytestmark = pytest.mark.parity
LIB = Path(nn.__file__).parent

@pytest.fixture(scope='module')
def weights():
    with np.load(LIB / 'needle_graphnet.npz', allow_pickle=False) as data:
        return dict(data)


def session(name):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(LIB / name), options, providers=['CPUExecutionProvider'])


@pytest.mark.parametrize('kernel,dilation', [(1,1), (3,1), (3,2), (3,4), (5,1), (5,4)])
def test_numpy_convolution_matches_torch(kernel, dilation):
    import torch
    rng = np.random.default_rng(17)
    x = rng.normal(size=(3,91)).astype('float32')
    w = rng.normal(size=(5,3,kernel)).astype('float32')
    b = rng.normal(size=5).astype('float32')
    expected = torch.nn.functional.conv1d(torch.from_numpy(x[None]), torch.from_numpy(w), torch.from_numpy(b),
                                         padding=(kernel//2)*dilation, dilation=dilation).numpy()[0]
    np.testing.assert_allclose(nn._conv1d_same_np(x,w,b,dilation), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('size', [1, 4, 9])
def test_tip_bundle_onnx_matches_numpy(weights, size):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(size,26)).astype('float32')
    edges = rng.normal(size=(size,size,8)).astype('float32')
    mask = np.ones((size,size), dtype='float32')
    actual = session('tip_aware_bundle_graph_transformer.onnx').run(None, {'x_26':x,'adj_feat_8':edges,'adj_mask':mask})[0]
    np.testing.assert_allclose(actual, nn._forward_tip_bgt_numpy(x,edges,mask,weights), rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize('seed', [0, 42, 107])
def test_tip_localizer_onnx_matches_numpy(weights, seed):
    x = np.random.default_rng(seed).normal(size=(24,91)).astype('float32')
    dz, probs, std = session('dedicated_1d_tip_localizer.onnx').run(None, {'profile_24x91':x[None]})
    expected = nn._forward_1d_tip_localizer_numpy(x, weights)
    np.testing.assert_allclose([dz[0],std[0],probs.max()], expected, rtol=1e-4, atol=2e-5)
    np.testing.assert_allclose(probs.sum(), 1, atol=1e-6)


def test_slice_mpnn_onnx_matches_numpy(weights):
    rng = np.random.default_rng(42)
    feed = dict(nodes_feat=rng.normal(size=(6,16)).astype('float32'),
                edge_feat=rng.normal(size=(5,12)).astype('float32'),
                src=np.arange(5,dtype='int64'), dst=np.arange(1,6,dtype='int64'),
                sp_src=np.array([0,1,2],dtype='int64'), sp_dst=np.array([2,3,4],dtype='int64'),
                sp_feat=rng.normal(size=(3,5)).astype('float32'))
    actual = session('needle_slice_mpnn.onnx').run(None,feed)
    expected = nn._forward_mpnn_numpy(**feed,W=weights)
    for a,e in zip(actual, expected): np.testing.assert_allclose(a,e,rtol=1e-4,atol=2e-5)


def test_unet_onnx_shape_and_finite_output():
    x = np.random.default_rng(42).normal(size=(1,6,16,32,32)).astype('float32')
    output = session('needle_unet3d.onnx').run(None, {'input':x})[0]
    assert output.shape == (1,5,16,32,32)
    assert np.isfinite(output).all()

@pytest.mark.parametrize('batch', [1, 3])
def test_tip_export_matches_actual_training_architecture(batch):
    import torch
    from tools.export_tip_model import load_model
    x = np.random.default_rng(9).normal(size=(batch,24,91)).astype('float32')
    with torch.no_grad(): expected = load_model()(torch.from_numpy(x))
    actual = session('dedicated_1d_tip_localizer.onnx').run(None, {'profile_24x91':x})
    for a,e in zip(actual,expected):
        np.testing.assert_allclose(a,e.numpy(),rtol=1e-4,atol=2e-5)
