# Azure Setup

> Author: Bakhtiyar Khan · Date: 2026-06-27

Step-by-step provisioning of every Azure resource the pipeline needs. All
commands use the Azure CLI (`az`). Replace placeholder values as appropriate.

## 0. Prerequisites

- An Azure subscription with Owner/Contributor rights.
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) installed
  and logged in: `az login`.
- The Azure ML CLI extension: `az extension add -n ml`.

```bash
# Common variables (edit these)
export LOCATION="eastus"
export RG="mlops-rg"
export WS="mlops-ws"
export STORAGE="mlopsstore$RANDOM"
export CONTAINER="mlops-data"
export ACR="mlopsacr$RANDOM"
export AKS="mlops-aks"
export SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
```

## 1. Resource group

```bash
az group create --name "$RG" --location "$LOCATION"
```

## 2. Azure Machine Learning workspace

```bash
az ml workspace create --name "$WS" --resource-group "$RG" --location "$LOCATION"
```

## 3. Azure Blob Storage

```bash
az storage account create --name "$STORAGE" --resource-group "$RG" \
  --location "$LOCATION" --sku Standard_LRS
az storage container create --name "$CONTAINER" --account-name "$STORAGE"
```

## 4. Azure Container Registry

```bash
az acr create --name "$ACR" --resource-group "$RG" --sku Basic --admin-enabled true
```
> Admin user is enabled so ACI can pull images with username/password.

## 5. Azure Kubernetes Service

```bash
az aks create --name "$AKS" --resource-group "$RG" --node-count 2 \
  --generate-ssh-keys --attach-acr "$ACR"
```

## 6. Application Insights (monitoring)

```bash
az extension add -n application-insights
az monitor app-insights component create --app mlops-ai \
  --location "$LOCATION" --resource-group "$RG"
# Copy the connectionString from the output into APPINSIGHTS_CONNECTION_STRING
```

## 7. Service principal for GitHub Actions

```bash
az ad sp create-for-rbac --name "mlops-ci" \
  --role contributor \
  --scopes "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RG" \
  --sdk-auth
```
The output contains `clientId`, `clientSecret`, `tenantId`, `subscriptionId` →
map these to the `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`,
`AZURE_SUBSCRIPTION_ID` GitHub Secrets.

## 8. Required GitHub Secrets

| Secret | Source |
|--------|--------|
| `AZURE_CLIENT_ID` | service principal |
| `AZURE_CLIENT_SECRET` | service principal |
| `AZURE_TENANT_ID` | service principal |
| `AZURE_SUBSCRIPTION_ID` | `az account show` |
| `AZURE_RESOURCE_GROUP` | `$RG` |
| `AZURE_WORKSPACE_NAME` | `$WS` |
| `AZURE_STORAGE_ACCOUNT` | `$STORAGE` |
| `AZURE_STORAGE_CONTAINER` | `$CONTAINER` |
| `AZURE_ACR_NAME` | `$ACR` |
| `AZURE_AKS_CLUSTER` | `$AKS` |
| `OPENROUTER_API_KEY` | https://openrouter.ai |
| `APPINSIGHTS_CONNECTION_STRING` | Application Insights component |

Add them via GitHub → Settings → Secrets and variables → Actions, or:

```bash
gh secret set AZURE_RESOURCE_GROUP --body "$RG"
# ...repeat for each secret
```

## 9. Teardown (avoid charges)

```bash
az group delete --name "$RG" --yes --no-wait
```
