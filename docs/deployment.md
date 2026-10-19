# Deployment (Azure Container Apps)

The API runs on Azure Container Apps (UK South), pulling a public image from
GitHub Container Registry. Scale-to-zero keeps idle cost near zero.

**Live URL:** https://carbon-forecast-api.delightfulfield-1ac7d123.uksouth.azurecontainerapps.io
(`/health`, `/forecast`, `/model`, `/docs`)

## One-time setup

```powershell
az login --tenant <TENANT_ID>        # browser sign-in (security defaults block device code)
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az group create --name rg-carbon-forecast --location uksouth
az containerapp env create --name cae-carbon-forecast --resource-group rg-carbon-forecast --location uksouth
```

A $10/month budget with alerts at 50%, 90% (actual) and 100% (forecasted) guards spending.

## Build and publish an image

Images are tagged with the champion model version, so every deployment maps to exactly one model.

```powershell
docker build -t gb-carbon-forecast:local .
docker tag gb-carbon-forecast:local ghcr.io/vijendrapokharkar15-design/gb-carbon-forecast:<MODEL_VERSION>
docker push ghcr.io/vijendrapokharkar15-design/gb-carbon-forecast:<MODEL_VERSION>
```

## Create the app (first deployment)

```powershell
az containerapp create `
  --name carbon-forecast-api --resource-group rg-carbon-forecast `
  --environment cae-carbon-forecast `
  --image ghcr.io/vijendrapokharkar15-design/gb-carbon-forecast:<MODEL_VERSION> `
  --target-port 8000 --ingress external `
  --cpu 0.5 --memory 1.0Gi --min-replicas 0 --max-replicas 1
```

## Deploy a new version, or roll back

Same command either way: point the app at the image tag you want.

```powershell
az containerapp update --name carbon-forecast-api --resource-group rg-carbon-forecast `
  --image ghcr.io/vijendrapokharkar15-design/gb-carbon-forecast:<MODEL_VERSION>
```

## Operations

```powershell
az containerapp logs show --name carbon-forecast-api --resource-group rg-carbon-forecast --follow
az containerapp show --name carbon-forecast-api --resource-group rg-carbon-forecast --query properties.latestRevisionName
```

## Shut everything down

Deletes the app, environment and logs in one go:

```powershell
az group delete --name rg-carbon-forecast --yes
```

## Notes

- First request after idle takes 20 to 60 s (cold start from zero replicas)
- `/forecast` always serves the latest 12:00 UTC issue; the daily recorder freezes it at 12:10 UTC