#define     DIM    256
#define     DT    .1


///////////////////////////////////////
// CUDA Kernel
///////////////////////////////////////
__device__ int checkPosition(int _pos){
  int dmax = DIM*DIM;
  if (_pos < 0){
    // _pos = dmax+_pos;
    _pos += DIM;
    // return _pos % dmax;
    return _pos;
  }
  else return _pos % dmax;
}

__global__ void RunOnce( float4 *_chem) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  _chem[offset] = make_float4(0.,0.,0.,1.);
}

__global__ void DrawSquare( float4 *_chem ) {
  if (threadIdx.x > DIM || threadIdx.y > DIM) return;

  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + (y * blockDim.x * gridDim.x);

  // q1. draws a square
  float posX = (float)x/DIM;
  float posY = (float)y/DIM;
  // if ( x < 140 && x > 116 && y < 140 && y > 116 ) {
  if ( x < 200 && x > 116 && y < 140 && y > 30 ) {
  // if ( posX < .75 && posX > .45 && posY < .55 && posY > .45 ) {
  // if ( posX < m_x+.05 && posX > m_x-.05 && posY < m_y+.05 && posY > m_y-.05 ) {    //use mouse position
    _chem[offset] = make_float4(1.,1.,1.,1.);
  }

}

__global__ void Diffusion( float4 *_chem, float4 *_lap, float _difConst, int mouse_x, int mouse_y) {
  if (threadIdx.x > DIM || threadIdx.y > DIM) return;

  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + (y * blockDim.x * gridDim.x);

  // constants
  // float xLength = (float)DIM/100.0;
  float xLength = 2.56;
  // float dx = (float)xLength/DIM;
  float dx = 0.01;
  float alpha = _difConst * DT / (dx*dx);

  // int n1 = (x+1)%DIM;
  // int n2 = (x-1)%DIM;
  // int n3 = (y+1)%DIM;
  // int n4 = (y-1)%DIM;

  // if (n2 < 0) n2 += DIM;
  // if (n4 < 0) n4 += DIM;

  // n1 = ((n1 + y * blockDim.x * gridDim.x)) % (DIM*DIM);
  // n2 = ((n2 + y * blockDim.x * gridDim.x)) % (DIM*DIM);
  // n3 = ((x + n3 * blockDim.x * gridDim.x)) % (DIM*DIM);
  // n4 = ((x + n4 * blockDim.x * gridDim.x)) % (DIM*DIM);
  // if (n1 > (y*DIM)+DIM) n1 -= DIM;

  int n1 = offset + 1;
  int n2 = offset - 1;
  int n3 = offset + DIM;
  int n4 = offset - DIM;

  if (n1 > ((DIM-1) + (y * blockDim.x * gridDim.x))) n1 -= DIM;
  if (n1 >= DIM*DIM) n1 -= DIM;

  if (n2 < (0 + (y * blockDim.x * gridDim.x))) n2 = ((DIM-1) + (y * blockDim.x * gridDim.x));
  if (n2 < 0) n2 += DIM;

  if (n3 >= DIM*DIM) n3 = x;

  if (n4 < 0) n4 = (DIM*DIM) - DIM + x;

  // int n1 = checkPosition((x+1) + y * blockDim.x * gridDim.x);
  // int n2 = checkPosition((x-1) + y * blockDim.x * gridDim.x);
  // int n3 = checkPosition(x + (y+1) * blockDim.x * gridDim.x);
  // int n4 = checkPosition(x + (y-1) * blockDim.x * gridDim.x);
  // __syncthreads();

  _lap[offset] = -4.0f * _chem[offset] + _chem[n1] + _chem[n2] + _chem[n3] + _chem[n4];
  _lap[offset] *= alpha;

}

__global__ void AddLaplacian( float4 *_chem, float4 *_lap) {
  if (threadIdx.x > DIM || threadIdx.y > DIM) return;

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  _chem[offset] += _lap[offset];
  _chem[offset].w = 1.0;

}

__global__ void React( float4 *_chemA, float4 *_chemB, float4 *_rA, float4 *_rB) {
  // if (threadIdx.x > DIM || threadIdx.y > DIM) return;

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float F = 0.05;
  float k = 0.0675;
  float4 A = _chemA[offset];
  float4 B = _chemB[offset];

  float4 reactionA = make_float4(-A.x * (B.x*B.x) + (F * (1.0-A.x)),
                                -A.y * (B.y*B.y) + (F * (1.0-A.y)),
                                -A.z * (B.z*B.z) + (F * (1.0-A.z)),
                                -A.w * (B.w*B.w) + (F * (1.0-A.w))
                                );

  float4 reactionB = make_float4(A.x * (B.x*B.x) - (F+k)*B.x,
                                A.y * (B.y*B.y) - (F+k)*B.y,
                                A.z * (B.z*B.z) - (F+k)*B.z,
                                A.w * (B.w*B.w) - (F+k)*B.w
                                );

  _rA[offset] = reactionA * .1;
  _rB[offset] = reactionB * .1;

  // _chemA[offset] += (DT * reactionA); //need parenthesis
  // _chemA[offset].w = 1.0;

  // _chemB[offset] += (DT * reactionB);
  // _chemB[offset].w = 1.0;
}

__global__ void AddReaction( float4 *_chemA, float4 *_chemB, float4 *_rA, float4 *_rB) {
  // if (threadIdx.x > DIM || threadIdx.y > DIM) return;

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  _chemA[offset] += _rA[offset];
  _chemA[offset].w = 1.0;

  _chemB[offset] += _rB[offset];
  _chemB[offset].w = 1.0;
}
