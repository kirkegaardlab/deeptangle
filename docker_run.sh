if [ ! -f celegans_512.avi ]
then
	wget https://sid.erda.dk/share_redirect/c3n4DHbGdI -O celegans_512.avi
fi
docker build --tag deeptangle_cpu .
docker run -v $(pwd):/mnt deeptangle_cpu
