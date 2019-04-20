#define NDEBUG

#include <iostream>
#include <string>
#include <fstream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "constants.h"
#include "sph_solver.h"

using namespace std;
using namespace Constants;

#define WINDOW_TITLE_PREFIX "SPH demo"

void display();
void init();
void timer(int);

SPHSolver sph;
float interval = TIMESTEP * 1000 * 100;
int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(RENDER_WIDTH, RENDER_HEIGHT);
	glutCreateWindow(WINDOW_TITLE_PREFIX);
	init();
	glutDisplayFunc(display);
	glutTimerFunc(interval, timer, 0);
	glutMainLoop();

	return 0;

	//cout << "---------------WRITE FRAME DATA------------" << endl;
	//std::ofstream ofs("../../canvas/out/data/test.json");
	//ofs << sph.data.dump();
	//ofs.close();
}

void init()
{
	cout << "---------------DEMO SPH START------------" << endl;
	sph = SPHSolver();

	glClearColor(0, 0, 0, 0); // moved this line to be in the init function
	glDisable(GL_DEPTH_TEST);

	// next four lines are new
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, RENDER_WIDTH - 1, RENDER_HEIGHT - 1, 0, -1.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
}

void DrawCircle(float cx, float cy, float radius) {
	int i, n = 100;
	float x, y;
	double rate;
	glColor3f(1.0, 1.0, 1.0); 
	glBegin(GL_POLYGON);
	for (i = 0; i < n; i++) {
		rate = (double)i / n;
		x = radius * cos(2.0 * M_PI * rate) + cx;
		y = radius * sin(2.0 * M_PI * rate) + cy;
		glVertex3f(x, y, 0.0); 
	}
	glEnd();
}

void timer(int value) {

	sph.update(TIMESTEP);
	glutPostRedisplay();
	glutTimerFunc(interval, timer, 0);
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();	
	for (int i = 0; i < sph.numberParticles; i++) {
		sph.particles[i].force = Vector2<float>(0.0f, 0.0f);
		DrawCircle(sph.particles[i].position.x* SCALE, sph.particles[i].position.y* SCALE, 0.5f * PARTICLE_SPACING * SCALE);
	}
	
	glFlush();
}

