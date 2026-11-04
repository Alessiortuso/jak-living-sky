# Dark Jak's horns (in Jak's neckB joint's own space) and a claw (along +z), as glb for build-actor
import json, struct, io, math, sys
import numpy as np
from PIL import Image
ORIG=sys.argv[1]; OUT=sys.argv[2]
CT={5120:'b',5121:'B',5122:'h',5123:'H',5125:'I',5126:'f'}; NC={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
b=open(ORIG,'rb').read(); l=struct.unpack('<I',b[12:16])[0]; j=json.loads(b[20:20+l]); off=20+l; bl=struct.unpack('<I',b[off:off+4])[0]; bin_=b[off+8:off+8+bl]
def acc(i):
    a=j['accessors'][i]; bv=j['bufferViews'][a['bufferView']]; n=NC[a['type']]; dt=np.dtype(CT[a['componentType']])
    return np.frombuffer(bin_,dtype=dt,count=a['count']*n,offset=bv.get('byteOffset',0)+a.get('byteOffset',0)).reshape(a['count'],n)
skin=j['skins'][0]; jn=[j['nodes'][k]['name'] for k in skin['joints']]; ibm=acc(skin['inverseBindMatrices']).reshape(-1,4,4)
IB={n:ibm[k].T for k,n in enumerate(jn)}  # model -> joint space

def tube(path, radii, sides=14, ridge=None):
    """vertices, normals, uvs, indices of a tube along path (Nx3) with radii (N)"""
    P=np.array(path); N=len(P); V=[];Nm=[];UV=[];I=[]
    T=np.gradient(P,axis=0); T/=np.linalg.norm(T,axis=1)[:,None]
    ref=np.array([0,0,1.0]) if abs(T[0][2])<0.9 else np.array([1.0,0,0])
    n0=np.cross(T[0],ref); n0/=np.linalg.norm(n0)
    frames=[]; n=n0
    for i in range(N):
        n=n-np.dot(n,T[i])*T[i]; n/=np.linalg.norm(n); bb=np.cross(T[i],n); frames.append((n,bb))
    for i in range(N):
        n,bb=frames[i]; r=radii[i]
        for s in range(sides+1):
            a=2*math.pi*s/sides; d=math.cos(a)*n+math.sin(a)*bb
            V.append(P[i]+d*r); Nm.append(d); UV.append((s/sides, i/(N-1)))
    for i in range(N-1):
        for s in range(sides):
            a=i*(sides+1)+s; c=a+sides+1
            I+=[a,c,a+1, a+1,c,c+1]
    # cap the base
    base=len(V); V.append(P[0]); Nm.append(-T[0]); UV.append((0.5,0.0))
    for s in range(sides): I+=[base, s+1, s]
    return np.array(V,np.float32),np.array(Nm,np.float32),np.array(UV,np.float32),np.array(I,np.uint32)

def bez(p0,p1,p2,p3,n):
    t=np.linspace(0,1,n)[:,None]; return (1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t**2*p2+t**3*p3

def to_joint(V,Nm,joint):
    M=IB[joint]; Vh=np.c_[V,np.ones(len(V))]@M.T; Nr=Nm@M[:3,:3].T; Nr/=np.linalg.norm(Nr,axis=1)[:,None]
    return Vh[:,:3].astype(np.float32),Nr.astype(np.float32)

def write_glb(path, parts, tex_png):
    # parts: list of (V,N,UV,I)
    V=np.concatenate([p[0] for p in parts]); Nm=np.concatenate([p[1] for p in parts]); UV=np.concatenate([p[2] for p in parts])
    I=[]; o=0
    for p in parts: I.append(p[3]+o); o+=len(p[0])
    I=np.concatenate(I).astype(np.uint32)
    blobs=[V.tobytes(),Nm.tobytes(),UV.tobytes(),I.tobytes(),tex_png]
    views=[];data=b''
    for k,bl_ in enumerate(blobs):
        while len(data)%4: data+=b'\0'
        views.append({'buffer':0,'byteOffset':len(data),'byteLength':len(bl_)}); data+=bl_
    while len(data)%4: data+=b'\0'
    for k in range(3): views[k]['target']=34962
    views[3]['target']=34963
    accs=[{'bufferView':0,'componentType':5126,'count':len(V),'type':'VEC3','min':V.min(0).tolist(),'max':V.max(0).tolist()},
          {'bufferView':1,'componentType':5126,'count':len(V),'type':'VEC3'},
          {'bufferView':2,'componentType':5126,'count':len(V),'type':'VEC2'},
          {'bufferView':3,'componentType':5125,'count':len(I),'type':'SCALAR'}]
    import os; name=os.path.basename(path)[:-4]
    g={'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
       'nodes':[{'name':name,'mesh':0}],
       'meshes':[{'name':name,'primitives':[{'attributes':{'POSITION':0,'NORMAL':1,'TEXCOORD_0':2},'indices':3,'material':0}]}],
       'materials':[{'name':name+'-tex','pbrMetallicRoughness':{'baseColorTexture':{'index':0},'metallicFactor':0.0,'roughnessFactor':0.6}}],
       'textures':[{'source':0,'sampler':0}],'samplers':[{'magFilter':9729,'minFilter':9987,'wrapS':10497,'wrapT':33071}],
       'images':[{'bufferView':4,'mimeType':'image/png','name':name+'-tex'}],
       'buffers':[{'byteLength':len(data)}],'bufferViews':views,'accessors':accs}
    js=json.dumps(g).encode(); 
    while len(js)%4: js+=b' '
    out=b'glTF'+struct.pack('<II',2,12+8+len(js)+8+len(data))+struct.pack('<I4s',len(js),b'JSON')+js+struct.pack('<I4s',len(data),b'BIN\0')+data
    open(path,'wb').write(out)

def png(img):
    bio=io.BytesIO(); img.save(bio,'PNG'); return bio.getvalue()

# --- horns texture: along v (0 base .. 1 tip): dark slate with lighter ridged rings, tip bone-grey
W,H=32,128
t=np.zeros((H,W,3))
for y in range(H):
    v=y/(H-1); ring=0.5+0.5*math.cos(v*2*math.pi*9)
    base=np.array([38,38,48])*(1-v)+np.array([120,118,126])*v
    c=base*(0.75+0.35*ring)
    t[y,:]=c
noise=np.random.default_rng(1).random((H,W,1))*14
horn_png=png(Image.fromarray(np.clip(t+noise,0,255).astype(np.uint8)))
# --- claw texture: black, a glossy streak
t=np.zeros((64,32,3))
for y in range(64):
    for x in range(32):
        s=math.exp(-((x/31-0.3)/0.12)**2)
        t[y,x]=np.array([14,12,18])+np.array([90,88,100])*s*(0.4+0.6*y/63)
claw_png=png(Image.fromarray(np.clip(t,0,255).astype(np.uint8)))

# horns: on top of the head, up and curving back, a little outward (model space, then into neckB's space)
parts=[]
for sx in (1,-1):
    p0=np.array([0.11*sx,2.74,0.10]); p1=np.array([0.15*sx,3.02,0.12]); p2=np.array([0.21*sx,3.30,-0.02]); p3=np.array([0.26*sx,3.40,-0.26])
    path=bez(p0,p1,p2,p3,22)
    k=np.linspace(0,1,22)
    radii=0.058*(1-k)**0.8+0.004
    V,Nm,UV,I=tube(path,radii)
    V,Nm=to_joint(V,Nm,'neckB')
    if sx<0:  # (winding kept: the tube's own)
        pass
    parts.append((V,Nm,UV,I))
write_glb(OUT+'/darkjak-horns.glb',parts,horn_png)
# a claw along +z (its own space), curving down (-y), 0.11 long
path=np.array([[0,-0.012*(k**2),0.11*k] for k in np.linspace(0,1,14)])
k=np.linspace(0,1,14); radii=0.019*(1-k)**0.9+0.0015
write_glb(OUT+'/darkjak-claw.glb',[tube(path,radii,sides=10)],claw_png)
print('ok')
