# 🧪 Testing Guide

## 1. Watch the Demo
### 🎬 Watch the Demo
Watch demo.mp4 in repo or [▶️ Click here to watch the demo](https://wh1522657.ispot.cc/demo.mp4)

Also watch the Set up in setup.mp4 or [▶️ Click here to watch the setup](https://wh1522657.ispot.cc/setup.mp4)

> Note: GitHub Markdown preview does not reliably render the HTML `<video>` tag, so this link is provided as a working fallback.

---

## 2. Test Workflow Overview
This application includes multiple test scenarios, progressing from simple to more advanced.

### Step 1: Authentication
Sign in to both GitHub and Microsoft using the following credentials:

- **Email:** eece00@outlook.com  
- **Password:** Poiuyt$12345678  

---

### Step 2: Trigger an Error
1. Go to the repository:  
   https://github.com/cig00/helper1  

2. Make changes to the app that intentionally cause an error.  
3. Commit your changes to the **master branch**.  

After committing, the app should automatically deploy to:  
https://wh1522657.ispot.cc/

---

### Step 3: Access the Deployed App
Log in using:

- **Email:** carlosgerges650@outlook.com  
- **Password:** Poiuyt$1234  

Navigate to the page where you introduced the error.

---

### Step 4: Verify Error Tracking
1. Open the **JIRA Dashboard**.  
2. Confirm that:
   - A new task has been created  
   - The error count in the summary has increased by **1**

---

### Step 5: Review Suggested Fix
1. Check GitHub for a **Pull Request (PR)**.  
2. You should see an automatically suggested fix.  
3. Review and merge the PR if it is correct.  

After merging, verify that the web app is functioning again.

---

## 3. Local Setup

### Clone and Run the App
1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip app.py
```
This will launch the Azure system setup locally.
Credentials are automatically configured.
---

## 4. Features

### 🔹 Auto Label Data
- Select a dataset file and a model  
- The system will automatically label the data  
- You can optionally download the latest log file from the Helper dashboard  

---

### 🔹 Train a Model
1. Choose the prepared dataset (from the previous step)  
2. Select where to run training:
   - **Local** → runs on your machine  
   - **Azure** → trains and deploys using Azure ML Studio  
3. Start the training process  
4. Make sure you're signed in to microsoft and enter those
SUBID: 64f9c97f-cd8e-4408-812c-d8fd13c0adf1
Tenant ID: a02e95d4-6401-4ba3-966e-0f469e1cced3
---

### 🔹 Host a New Model
- All required fields are pre-configured (because there's a lot of fields)  
- Select the model version you want to deploy  
- The system will automatically create a Pull Request (PR) in the Helper repository  
- Once merged, the application will start sending data to the newly deployed model  

**Important Note:** You may use your own accounts, as this app works with any account. However, we’ve tried to make the process as easy as possible for you. This app is just the set up
Your realy service is hosted on Azure.


# Thank you!

## If you have any questions, please get in touch with us — we’ll be happy to help.