# rumor-tracker-uganda-covid
Scripts to group rumors and fake news about COVID-19 into topics.

Built to support Uganda Red Cross Society (URCS) during COVID-19 response. 

## Introduction
Rumors and fake news on COVID-19 are being collected by volunteers of URCS around the country. This repo contains the code to build & run a model which:
1. Groups short messages: rumors and fake news
2. Assign a topic and a representative example to each group

Built on top of [GSDMM: short text clustering](https://github.com/rwalk/gsdmm).

N.B. the creation of groups (a.k.a. clustering) is automated, but the topic description is not. You need a human to read some representative examples of each group and come up with a meaningful, human-readable description.

## Setup and common issues
* Sync repository locally
* Make sure that openSSL is installed (https://slproweb.com/products/Win32OpenSSL.html)
* Install the right libraries. Among others you need spellchecker and indexer. Indexer is install through pyspellchecker (https://stackoverflow.com/questions/57602566/pip-install-indexer-error-in-python-3-7-in-windows-10)
* Update google-api-python-client (https://stackoverflow.com/questions/45477016/importerror-cannot-import-name-discovery).
* Make sure that you have the right service_account_key.json (see below) and that the account related to the service account has access to the google sheets (https://stackoverflow.com/questions/38949318/google-sheets-api-returns-the-caller-does-not-have-permission-when-using-serve)

## Usage
Rumors (input) and topics (output) are stored in [Google Sheets](https://docs.google.com/spreadsheets/d/18PwsExSVerYzTxGxarLwyGkKIVT2QJCobCnoeLYXwjM/edit#gid=0), access credentials are stored in `service_account_key.json`, ask Jacopo Margutti (jmargutti@redcross.nl).

Run the model with
```
python run_model.py
```
