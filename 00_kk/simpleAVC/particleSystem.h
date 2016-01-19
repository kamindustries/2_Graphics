#include "particle.h"
#include "MERSENNE_TWISTER.h"

#define MAX_PARTICLES 10000

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

  void updateAndDraw(float *_u, float *_v, int arraySize, float win_size){
    float *u_cpu = (float*)malloc(sizeof(float)*arraySize);
    float *v_cpu = (float*)malloc(sizeof(float)*arraySize);
    checkCudaErrors(cudaMemcpy(u_cpu, _u, sizeof(float)*arraySize, cudaMemcpyDeviceToHost ));
    checkCudaErrors(cudaMemcpy(v_cpu, _v, sizeof(float)*arraySize, cudaMemcpyDeviceToHost ));

    glEnable(GL_BLEND);
  	glDisable(GL_TEXTURE_2D);
    glBlendFunc(GL_ONE,GL_ONE);
    glLineWidth(1);

    for (int i = 0; i < MAX_PARTICLES; i++){
      if (particles[i].alpha > 0){
        // particles[i].update(u_cpu, v_cpu, win_size);
        particles[i].updateVertexArrays(i, win_size, posArray, colArray);
      }
    }

    glEnableClientState(GL_VERTEX_ARRAY);
  	glVertexPointer(2, GL_FLOAT, 0, posArray);

  	glEnableClientState(GL_COLOR_ARRAY);
  	glColorPointer(3, GL_FLOAT, 0, colArray);

  	glDrawArrays(GL_LINES, 0, MAX_PARTICLES * 2);

  	glDisableClientState(GL_VERTEX_ARRAY);
  	glDisableClientState(GL_COLOR_ARRAY);

  	glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);

    free(u_cpu);
    free(v_cpu);


  }
  void addParticle(float x, float y){
    particles[curIndex].init(x, y);
    curIndex++;
    if (curIndex >= MAX_PARTICLES) curIndex = 0;
  }
  void addParticles(float x, float y, int count){
    for (int i = 0; i < count; i++){
      // addParticle(x+(randNum.rand() * 5.0f), y+(randNum.rand() * 5.0f));
      addParticle(x, y);
    }
  }

};
