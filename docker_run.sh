if [ ! -f celegans_512.avi ]
then
	wget https://sid.erda.dk/share_redirect/c3n4DHbGdI -O celegans_512.avi
fi
docker build --tag deeptangle_cpu .
echo "----------------------------------------------------------------"
echo "Running detection on single frame (note: on CPU = slow)  [detect.py]"
echo "----------------------------------------------------------------"
docker run -v $(pwd):/mnt deeptangle_cpu python3 examples/detect.py --model=weights/ --input=/mnt/celegans_512.avi --output=/mnt/output.png --frame=100 --correction_factor=1.2
echo ""
echo "----------------------------------------------------------------"
echo "Running tracking on 50 frames (note: on CPU = slow)  [track.py]"
echo "----------------------------------------------------------------"
docker run -v $(pwd):/mnt deeptangle_cpu python3 examples/track.py --model=weights/ --input=/mnt/celegans_512.avi --output=/mnt/track/ --correction_factor=1.2 --num_frames=50

