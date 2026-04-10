# House Segmentation Service

## Running the Container Locally
1. Build the image:
   `docker build -t house-segmentation .`
2. Run the container (injecting the secret environment variable):
   `docker run -p 5000:5000 -e API_SECRET_KEY=super_secret_lab2_key house-segmentation`

## Docker Hub Repository
[Link to your Docker Hub repository: https://hub.docker.com/r/lindensheehy/house-segmentation]

## Sample API Requests
**Health Check:**
`curl http://localhost:5000/health`

**Segmentation Prediction:**
`curl -X POST http://localhost:5000/predict -H "X-API-Key: super_secret_lab2_key" -F "file=@sample_aerial.jpg"`