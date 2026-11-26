import json, struct, numpy as np
CT={5126:np.float32,5125:np.uint32,5123:np.uint16,5121:np.uint8}
NC={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}
def read(path):
    d=open(path,'rb').read(); L=struct.unpack('<I',d[12:16])[0]; j=json.loads(d[20:20+L])
    o=20+L; BL=struct.unpack('<I',d[o:o+4])[0]; bin_=d[o+8:o+8+BL]
    def acc(i):
        a=j['accessors'][i]; bv=j['bufferViews'][a['bufferView']]; st=bv.get('byteOffset',0)+a.get('byteOffset',0)
        k=NC[a['type']]; dt=np.dtype(CT[a['componentType']]); stride=bv.get('byteStride',0)
        if stride and stride!=k*dt.itemsize:
            raw=np.frombuffer(bin_,dtype=np.uint8,count=stride*(a['count']-1)+k*dt.itemsize,offset=st)
            rows=np.lib.stride_tricks.as_strided(raw,shape=(a['count'],k*dt.itemsize),strides=(stride,1)).copy()
            arr=rows.view(dt).reshape(a['count'],k)
            return arr if k>1 else arr[:,0]
        arr=np.frombuffer(bin_,dtype=dt,count=a['count']*k,offset=st)
        return arr.reshape(a['count'],k) if k>1 else arr
    p=j['meshes'][0]['primitives'][0]; A=p['attributes']
    img=j['images'][0]; bv=j['bufferViews'][img['bufferView']]; tex=bin_[bv.get('byteOffset',0):bv.get('byteOffset',0)+bv['byteLength']]
    return dict(pos=acc(A['POSITION']).astype(np.float64), nrm=acc(A['NORMAL']).astype(np.float64) if 'NORMAL' in A else None, uv=acc(A['TEXCOORD_0']).astype(np.float64), idx=acc(p['indices']).astype(np.int64).reshape(-1,3), tex=tex, mime=img.get('mimeType'))
