resource "helm_release" "argo" {
  name       = "argo-workflows"
  repository = "https://argoproj.github.io/argo-helm"
  chart      = "argo-workflows"
  namespace  = "argo"

  create_namespace = true
}