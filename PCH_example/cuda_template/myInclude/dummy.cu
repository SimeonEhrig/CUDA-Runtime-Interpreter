int main(int argc, char const *argv[])
{
	int * device;
	bar_device<int><<<1,4>>>(device);
	return 0;
}