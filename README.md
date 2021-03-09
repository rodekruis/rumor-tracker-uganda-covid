# rumor-tracker-uganda-covid
Scripts to group rumors and fake news about COVID-19 into topics.

Built to support Uganda Red Cross Society (URCS) during COVID-19 response. 

## Introduction
Rumors and fake news on COVID-19 are being collected by volunteers of URCS around the country. This repo contains the code to build & run a model which:
1. Groups short messages: rumors and fake news
2. Assign a topic and a representative example to each group

Built on top of [GSDMM: short text clustering](https://github.com/rwalk/gsdmm).

N.B. the creation of groups (a.k.a. clustering) is automated, but the topic description is not. You need a human to read some representative examples of each group and come up with a meaningful, human-readable description.

## Usage
Rumors (input) and topics (output) are stored in [Google Sheets](https://docs.google.com/spreadsheets/d/18PwsExSVerYzTxGxarLwyGkKIVT2QJCobCnoeLYXwjM/edit#gid=0), access credentials are stored in `service_account_key.json`, ask Jacopo Margutti (jmargutti@redcross.nl).

Run the model with
```
python represent_simple.py
```
