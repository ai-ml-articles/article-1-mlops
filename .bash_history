ls
git init
touch .gitignore
nano .gitignore
git restore --staged .bash_history  # Remove it from staging
git checkout -- .bash_history  # Discard changes to this file
git status
git add .
git commit -m "Initial commit: MLOps housing price prediction setup"
git remote add origin https://github.com/ai-ml-articles/article-1-mlops.git
git push -u origin main
gcloud auth application-default login
gcloud config set project mlops-housing-project
gcloud config set compute/region us-central1
gcloud services enable aiplatform.googleapis.com
pip install google-cloud-aiplatform
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gsutil ls
nano train.py
python train.py --bucket-name your-bucket-name --model-filename model.pkl
pip install joblib
python train.py --bucket-name your-bucket-name --model-filename model.pkl
pip install sklearn
pip install --user scikit-learn
python train.py --bucket-name your-bucket-name --model-filename model.pkl
nano train.py
python train.py --bucket-name mlops-housing-bucket --model-filename model.pkl
truncate train.py
rm train.py
touch train.py
nano train.py
python train.py --bucket-name mlops-housing-bucket --model-filename model.pkl
gsutil ls gs://your-bucket-name/models/
gsutil ls gs://mlops-housing-bucket/models/
nano Dockerfile
gcloud auth configure-docker
docker build -t gcr.io/mlops-housing-project/mlops-housing-train:latest .
docker push gcr.io/YOUR_PROJECT_ID/mlops-housing-train:latest
docker push gcr.io/mlops-housing-project/mlops-housing-train:latest
gcloud services enable artifactregistry.googleapis.com
docker push gcr.io/mlops-housing-project/mlops-housing-train:latest
gcloud container images list
gcloud ai custom-jobs create   --region=us-central1   --display-name=mlops-housing-training   --python-package-uris=gs://mlops-housing-bucket/packages/train.py   --python-module=train   --args="--bucket-name=mlops-housing-bucket","--model-filename=model.pkl"   --image-uri=gcr.io/mlops-housing-project/mlops-housing-train:latest   --machine-type=n1-standard-4
tar -czvf train.tar.gz train.py
gsutil cp train.tar.gz gs://mlops-housing-bucket/packages/
gcloud ai custom-jobs create   --region=us-central1   --display-name=mlops-housing-training   --python-package-uris=gs://mlops-housing-bucket/packages/train.tar.gz   --python-module=train   --args="--bucket-name=mlops-housing-bucket","--model-filename=model.pkl"   --machine-type=n1-standard-4
gcloud ai custom-jobs create   --region=us-central1   --display-name=mlops-housing-training   --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,python-package-uris=gs://mlops-housing-bucket/packages/train.tar.gz,python-module=train   --args="--bucket-name=mlops-housing-bucket" --args="--model-filename=model.pkl"
gcloud ai custom-jobs create   --region=us-central1   --display-name=mlops-housing-training   --python-package-uris=gs://mlops-housing-bucket/packages/train.tar.gz   --python-module=train   --args="--bucket-name=mlops-housing-bucket" --args="--model-filename=model.pkl"   --worker-pool-spec=machine-type=n1-standard-4,replica-count=1
gcloud ai custom-jobs create   --region=us-central1   --display-name=mlops-housing-training   --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,executor-image-uri=us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-8:latest,local-package-path=gs://mlops-housing-bucket/packages/train.tar.gz   --args="--bucket-name=mlops-housing-bucket" --args="--model-filename=model.pkl"
gcloud ai custom-jobs create   --region=us-central1   --display-name=mlops-housing-training   --python-package-uris=gs://mlops-housing-bucket/packages/train.tar.gz   --python-module=train.py   --args="--bucket-name=mlops-housing-bucket" --args="--model-filename=model.pkl"   --worker-pool-spec=machine-type=n1-standard-4,replica-count=1
gcloud ai custom-jobs create   --region=us-central1   --display-name=mlops-housing-training   --python-package-uris=gs://mlops-housing-bucket/packages/train.tar.gz   --args="--bucket-name=mlops-housing-bucket" --args="--model-filename=model.pkl"   --worker-pool-spec=machine-type=n1-standard-4,replica-count=1
gcloud ai custom-jobs create   --region=us-central1   --display-name=mlops-housing-training   --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,executor-image-uri=us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest   --python-package-uris=gs://mlops-housing-bucket/packages/train.tar.gz   --args="--bucket-name=mlops-housing-bucket" --args="--model-filename=model.pkl"
gcloud ai custom-jobs create   --region=us-central1   --display-name=mlops-housing-training   --python-module=train
gcloud ai models list --region=us-central1
ls
rm Dockerfile
ls
nano register_model.py
python register_model.py
nano register_model.py
python register_model.py
service-1008473433714@gcp-sa-aiplatform.iam.gserviceaccount.com
gcloud projects add-iam-policy-binding mlops-housing-project   --member=serviceAccount:service-1008473433714@gcp-sa-aiplatform.iam.gserviceaccount.com   --role=roles/storage.objectViewer
gcloud projects get-iam-policy mlops-housing-project --format=json | grep storage
python register_model.py
gcloud ai models list --region=us-central1
git status
