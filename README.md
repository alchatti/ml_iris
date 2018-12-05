# ML.NET Iris Console App

A console app for ML.NET based on the implementation of `Iris Plants` using https://archive.ics.uci.edu/ml/machine-learning-databases/iris as **database**, with support for training & testing CSV input files

| You need the `Data` Folder in the same directory as the Application

## Technology Stack

- .NET Core 2.1
- ML.NET 0.8

## Commands

### Create

```sh
# New Console Project
dotnet new console -o ml_iris
# Install ML.NET
dotnet add package Microsoft.ML
```

### Publish

https://docs.microsoft.com/en-us/dotnet/core/rid-catalog

```sh
dotnet publish
# Self Contained with Targeted OS
dotnet publish --self-contained -r win10-x64 -o ./bin/Win10-x64
dotnet publish --self-contained -r linux-x64 -o ./bin/linux-x64
```
