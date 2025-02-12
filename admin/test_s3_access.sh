# Load environment variables from .env file
set -o allexport
source .env
set +o allexport
# Now you can use the AWS CLI with the refreshed credentials
aws s3 ls s3://$TEST_AWS_BUCKET_NAME