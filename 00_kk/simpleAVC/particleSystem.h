#include "particle.h"
#include "MERSENNE_TWISTER.h"

#define MAX_PARTICLES 100000

class ParticleSystem {
public:

  Particle particles[MAX_PARTICLES];
  MERSENNE_TWISTER randNum;
  int curIndex;
  float *posArray;
  float *colArray;

  ParticleSystem(){
    curIndex = 0;
    posArray = (float*)malloc(sizeof(float) * MAX_PARTICLES*2*2);
    colArray = (float*)malloc(sizeof(float) * MAX_PARTICLES*3*2);
  }

  void updateAndDraw(float *_u, float *_v, int arraySize, float win_size, int fieldSizeX, int fieldSizeY){
    float *u_cpu = (float*)malloc(sizeof(float)*arraySize);
    float *v_cpu = (float*)malloc(sizeof(float)*arraySize);
    checkCudaErrors(cudaMemcpy(u_cpu, _u, sizeof(float)*arraySize, cudaMemcpyDeviceToHost ));
    checkCudaErrors(cudaMemcpy(v_cpu, _v, sizeof(float)*arraySize, cudaMemcpyDeviceToHost ));

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    // glBlendFunc(GL_SRC_COLOR,GL_ONE_MINUS_SRC_COLOR); // probably need to change this
    glBlendFunc(GL_ONE,GL_ONE); // probably need to change this
    glLineWidth(1);
    glColor3f ( 1.0f, 1.0f, 1.0f );

    for (int i = 0; i < MAX_PARTICLES; i++){
      if (particles[i].alpha > 0){
        particles[i].update(u_cpu, v_cpu, win_size, fieldSizeX, fieldSizeY);
        particles[i].updateVertexArrays(i, win_size, posArray, colArray);
      }
    }

    glEnableClientState(GL_VERTEX_ARRAY);
  	glVertexPointer(2, GL_FLOAT, 0, posArray);

  	glEnableClientState(GL_COLOR_ARRAY);
  	glColorPointer(3, GL_FLOAT, 0, colArray);

  	// glDrawArrays(GL_POINTS, 0, MAX_PARTICLES * 2);
  	glDrawArrays(GL_LINES, 0, MAX_PARTICLES * 2);

  	glDisableClientState(GL_VERTEX_ARRAY);
  	glDisableClientState(GL_COLOR_ARRAY);

  	glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);

    free(u_cpu);
    free(v_cpu);

  }

  void addParticle(float x, float y){
    particles[curIndex].init(x, y, randNum);
    curIndex++;
    if (curIndex >= MAX_PARTICLES) curIndex = 0;
  }
  void addParticles(float x, float y, int count, float radius){
    for (int i = 0; i < count; i++){
      float rndx = (randNum.rand()*2.0)-1.0;
      // rndx *= radius;
      float rndy = (randNum.rand()*2.0)-1.0;
      // rndy *= radius;
      // addParticle(x+(rndx * radius), y+(rndx * radius));
      addParticle(.01*cos(rndx*2.0*3.1459) + x, .01*sin(rndx*2.0*3.1459) + y);
    }
  }

};
