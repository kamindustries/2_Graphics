class Particle{
public:
  float posX, posY;
  float radius;
  float alpha;
  float mass;
  float momentum;
  float fluid_force;

  void init(float x, float y) {
    posX = x;
    posY = y;
    radius = 5;
    alpha = 1.0; //make random .3-1
    mass = .5; //make random .1-1
    momentum = 0.5;
    fluid_force = 0.6;
  }

  void update(float *u, float *v) {
    if (alpha == 0)
      return;

    velX = u[posX] * (mass * fluid_force) * windowSize + velX * momentum;
    velY = v[posY] * (mass * fluid_force) * windowSize + velY * momentum;

    posX += velX;
    posY += velY;

    alpha *= 0.999f;
    if (alpha < 0.01f)
      alpha = 0;

  };
};
