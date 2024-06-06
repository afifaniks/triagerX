# Triager X

Triager X is a novel triaging framework designed to streamline the process of handling GitHub issues. By analyzing the title and description of a GitHub issue, Triager X recommends the most appropriate components and developers to address the issue. This tool can significantly reduce the time and effort needed to manage and assign issues in software projects.

## Features
- **Automatic Component Recommendation**: Suggests relevant components for each GitHub issue based on its content.
- **Developer Assignment**: Identifies and recommends developers best suited to handle the issue.
- **Efficient Issue Management**: Enhances productivity by automating the triaging process, allowing teams to focus on resolving issues rather than sorting them.

## Build Docker Image
To build the Docker image for Triager X, run the following command:

```shell
docker build -t triagerx .
```

## Load Docker Image
To build the Docker image for Triager X, run the following command:

```shell
docker load -i triagerx.tar
```

## Run Docker Container
To run the Docker container on CPU, use the following command:
### CPU
```shell
docker run --rm -p 8000:80 --name triagerx triagerx
```

To run the Docker container with GPU support, use the following command:
### GPU
```shell
docker run --gpus all --rm -p 8000:80 --name triagerx triagerx
```

## Example API Request
To get component and developer recommendations for a GitHub issue, make a POST request to the `/recommendation` endpoint. Here is an example using `curl`:

```shell
curl -X 'POST' \
  'http://127.0.0.1:8000/recommendation' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "issue_title": "Issue Title from GitHub",
  "issue_description": "Issue Description from GitHub"
}'
```

## Example API Response
The API will respond with a JSON object containing the recommended components and developers. Here is an example response:

```json
{
  "recommended_components": [
    "comp:vm",
    "comp:gc",
    "comp:test"
  ],
  "recommended_developers": [
    "pshipton",
    "keithc-ca",
    "babsingh"
  ]
}
```

## Swagger UI

You can also invoke the endpoint with Swagger UI.
To access the UI for using the API or reading the documentation,
navigate to the following address once the container is up and running:

```
http://127.0.0.1:8000/docs
```

## Usage
1. **Build the Docker Image**: Follow the instructions in the "Build Docker Image" section.
2. **Run the Docker Container**: Follow the instructions in the "Run Docker Container" section.
3. **Make API Requests**: Use the example API request to get recommendations for your GitHub issues.