#include "array_accessor1.h"
#include "point3.h"
#include "sph3_solver.h"

int main(int argc, char *argv[])
{
	//// array_accessor
	//// test foreach
	//int data[] = {4, 5, 3, 4, 6, 6};
	//ArrayAccessor<int, 1> acc(6, data);
	//acc.forEach([](int elem) {
	//    printf("foreach: %d\n", elem);
	//});

	////acc.forEachIndex([&](size_t i) {
	////	acc[i] = 4.f * i + 1.5f;
	////	printf("foreach index: %d\n", acc[i]);
	////});

	////acc.parallelForEach([](int& elem) {
	////	elem *= 2;
	////	printf("parallel foreach: %d\n", elem);
	////});

	////acc.parallelForEachIndex([&](size_t i) {
	////	 acc[i] *= 2;
	////	 printf("parallel foreach index: %d\n", acc[i]);
	////});

	//Point3F p3f1(3.4, 3.1, 4);
	//Point3F p3f2(0.6, 0.9, 0.0);
	//p3f1.iadd(p3f2);
	//Point3F p3f3 = p3f1.add(p3f2);
	//printf("Point3F p3f1: %f, %f, %f\n", p3f1.x, p3f1.y, p3f1.z);
	//printf("Point3F p3f3: %f, %f, %f\n", p3f3.x, p3f3.y, p3f3.z);


    return 0;
}