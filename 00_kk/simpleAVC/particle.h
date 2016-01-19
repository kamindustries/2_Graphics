class Particle{
public:
  float posX, posY, velX, velY;
  float radius;
  float alpha;
  float mass;
  float momentum;
  float fluid_force;

  void init(float x, float y) {
    posX = x;
    posY = y;
    velX = 0.0;
    velY = 0.0;
    radius = 5;
    alpha = 1.0; //make random .3-1
    mass = .5; //make random .1-1
    momentum = 0.5;
    fluid_force = 0.6;
  }

  void update(float *u, float *v, float windowSize) {
    if (alpha == 0.0) return;

    velX = u[int(posX)] * (mass * fluid_force) * windowSize + velX * momentum;
    velY = v[int(posY)] * (mass * fluid_force) * windowSize + velY * momentum;

    posX += velX;
    posY += velY;

    // alpha *= 0.999f;
    if (alpha < 0.01f)
      alpha = 0;
  };

  void updateVertexArrays( int i, float windowSize, float *posBuffer, float *colBuffer) {
  	int vi = i * 4;
      posBuffer[vi++] = posX - velX;
    	posBuffer[vi++] = posY - velY;
    	posBuffer[vi++] = posX;
    	posBuffer[vi++] = posY;

    int ci = i * 6;
  		colBuffer[ci++] = alpha;
  		colBuffer[ci++] = alpha;
  		colBuffer[ci++] = alpha;
  		colBuffer[ci++] = alpha;
  		colBuffer[ci++] = alpha;
  		colBuffer[ci++] = alpha;
  }
};
