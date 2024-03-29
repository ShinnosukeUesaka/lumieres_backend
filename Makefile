update-name:
	$(eval OLD_NAME := $(shell grep '^name = ' pyproject.toml | sed 's/name = "\(.*\)"/\1/'))
	@echo "Current project name is: $(OLD_NAME)"
	@echo "Enter new project name: "; \
	read NEW_NAME; \
	echo "Enter new GCP Project id: "; \
	read PROJECT_ID; \
	echo "Updating project name from $(OLD_NAME) to $$NEW_NAME..."; \
	mv $(OLD_NAME) $$NEW_NAME; \
	sed -i '' 's/$(OLD_NAME)/'$$NEW_NAME'/g' $$NEW_NAME/main.py; \
	sed -i '' 's/^name = "$(OLD_NAME)"/name = "'$$NEW_NAME'"/' pyproject.toml; \
	sed -i '' 's/IMAGE_NAME: $(OLD_NAME)/IMAGE_NAME: '$$NEW_NAME'/' .github/workflows/deploy.yml; \
	sed -i '' 's/SOURCE_FOLDER_NAME: $(OLD_NAME)/SOURCE_FOLDER_NAME: '$$NEW_NAME'/' .github/workflows/deploy.yml; \
	echo "Updating GCP Project id from  to $$PROJECT_ID..."; \
	sed -i '' 's/GCP_PROJECT_ID: .*/GCP_PROJECT_ID: '$$PROJECT_ID'/' .github/workflows/deploy.yml; \
	sed -i '' 's/$(OLD_NAME)/'$$NEW_NAME'/g' $$NEW_NAME/firebase_utils.py; 
