{{- define "heart-disease.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "heart-disease.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s" (include "heart-disease.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
