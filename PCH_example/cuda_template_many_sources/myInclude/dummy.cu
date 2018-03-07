int main(int argc, char const *argv[])
{
	int * device;
	foo_device<<<1,4>>>(device);
	add1<int><<<1,4>>>(device);
	sub1<int><<<1,4>>>(device);
	return 0;
}