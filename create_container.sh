gpu=0
while getopts "g:n:" OPTION; do
    case $OPTION in
        g) gpu=$OPTARG;;
		n) name=$OPTARG;;
        *) exit 1 ;;
    esac
done

image=adalmia/coreml:v1.0

if [[ -z $WANDB_API_KEY ]] ; then
	echo "ERROR: set the environment variable WANDB_API_KEY"
	exit 0
fi

NV_GPU=$gpu nvidia-docker run --rm -it \
	--name "$gpu"_"$name" \
	-v /path/to/coreml/:/workspace/coreml \
	-v /path/to/outputs/:/output \
	-v /path/to/data:/data \
	--env WANDB_DOCKER=$image \
	--env WANDB_API_KEY=$WANDB_API_KEY \
	--ipc host \
	$image bash
