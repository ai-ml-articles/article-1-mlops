ls
git init
touch .gitignore
nano .gitignore
git restore --staged .bash_history  # Remove it from staging
git checkout -- .bash_history  # Discard changes to this file
git status
