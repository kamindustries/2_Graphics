#include "MERSENNE_TWISTER.h"

class Particle{
public:
  float posX, posY, velX, velY;
  float radius;
  float alpha;
  float mass;
  float momentum;
  float fluid_force;

  void init(float x, float y, MERSENNE_TWISTER mt) {
    posX = x;
    posY = y;
    velX = 0.0;
    velY = 0.0;
    radius = 5;
    alpha = mt.rand(0.7f) + 0.3;
    mass = mt.rand(0.9f) + .1;
    momentum = 0.5;
    // fluid_force = 0.6;
    fluid_force = 5.0;
  }

  void update(float *u, float *v, float windowSize, int fieldSizeX, int fieldSizeY) {
    if (alpha < 0.0001) return;

    int px = floor(posX * 512);
    int py = floor(posY * 512);

    int id = ((px) + (512) * (py));
    if (id >= (512*512)-2) id = (512*512) - 2;
    if (id <= 1) id = 1;

    float h = 10.0f/windowSize; // not sure why i have to crank up to 10...
    velX = u[id] * (mass * fluid_force) * h + velX * momentum;
    velY = v[id] * (mass * fluid_force) * h + velY * momentum;

    posX += velX;
    posY += velY;

    if (posX < 0)
      posX = 0;
      velX *= -1.0;
    if (posY < 0)
      posY = 0;
      velY *= -1.0;

    if (posX > windowSize-1)
      posX = windowSize-1;
      velX *= -1.0;
    if (posY > windowSize-1)
      posY = windowSize-1;
      velY *= -1.0;

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
