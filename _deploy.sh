# configure your name and email if you have not done so
git config --global user.email "cimentadaj@gmail.com"
git config --global user.name "Jorge Cimentada"

# clone the repository to the book-output directory
git clone -b gh-pages \
  https://${GITHUB_PAT}@github.com/${TRAVIS_REPO_SLUG}.git \
  book-output

# Move to the bookoutput
cd book-output

# Remove everything there is there
git rm -rf *

# Create a figs directory
mkdir /figs

# Copy the docs file to the directoy
cp -r ../docs/* ./

# Copy the figures to the created figs folder
cp -r ../figs/* ./figs/

git add --all *
git commit -m "Update the book"
git push -q origin gh-pages
