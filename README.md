# Homework-upstream


Below there are a couple of instructions that should help you bring info from the upstream Homework directory into your repo. The assumption below is that you are in the directory that has the cloned version of your own 759 repo.		
		
		
	* Move your Makefile out of the way (into MakefileOld, for instance) and bring in a fresh copy
		```
		>> mv Makefile MakefileOld
		>> git checkout -- Makefile 
		```
	* Create a shortcut name for Tim’s remote repo. We’ll call it here upstream:
		```
		>> git remote add upstream https://github.com/UW-Madison-759/Homework-upstream.git
		```
	* Pull from repo nicknamed upstream what it has to offer (pulling from master branch)
		```
		>> git pull upstream master
		```
	* Do your work, then commit and push changes into origin repo (your own repo, that is)
		```
		>> git commit –m”synced w/ upstream; fixed bug in problem1.cpp”
		>> git push origin master
		```
## Notes
- If you don’t move Makefile out of the way, what you’ll get is a merge of the Makefile from upstream into your current Makefile. If this is what you want, then no need to mv and do the git checkout operation
- You are working w/ two repos: origin & upstream. The first is yours; the second is the one Tim put together 
- You need to sync w/ upstream every once in a while, just to make sure you collect the right homework assignment (which is placed in upstream), the latest Makefile, etc.
- If you have associated a public SSH key with your GitHub account, you can use
	```
	git remote add upstream git@github.com:UW-Madison-759/Homework-upstream.git
	```

