# rumor-tracker-uganda-covid
Scripts to group rumors about COVID-19 into topics.

Built to support Uganda Red Cross Society (URCS) during COVID-19 response. 

## Introduction
Rumors and fake news on COVID-19 are being collected by volunteers of URCS around the country. This repo contains the code to build & run a model which:
1. Groups short messages: rumors and fake news
2. Assign a topic and a representative example to each group

Built on top of [GSDMM: short text clustering](https://github.com/rwalk/gsdmm)

## Usage
rumors (input) and topics (output) are stored in [Google Ssheets](https://docs.google.com/spreadsheets/d/18PwsExSVerYzTxGxarLwyGkKIVT2QJCobCnoeLYXwjM/edit#gid=0), access credentials are stored in `service_account_key.json` (ask [Jacopo](jmargutti@redcross.nl)).

Run the model with
```
python represent_simple.py
```
