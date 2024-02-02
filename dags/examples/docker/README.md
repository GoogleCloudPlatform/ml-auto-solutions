This directory contains example build folders to demonstrate direct Docker
builds and custom builds using [DockerBuildTask](/xlml/apis/task.py#L45).

## Docker Build

A simple docker build requires only a Dockerfile in the target build path. The
DockerBuildTask can be used as follows to build a DAG-scoped image from the
`example_dockerfile` directory:

```python
from xlml.apis import task
from dags.vm_resource import DockerImage

docker_image_build = task.DockerBuildTask(
  build_dir="dags/examples/docker/example_dockerfile",
  image_name=DockerImage.XPK_JAX_TEST.value,
).run()

# The tagged image name is accessible via `docker_image_build.output`
```

## Custom Build

For more sophisticated builds or builds using Dockerfiles hosted outside of this
repository, a `cloudbuild.yaml` file can be used to specify the build. See the
[Cloud Build documentation](https://cloud.google.com/build/docs/build-config-file-schema)
for details on how to specify a config file.

To expose the generated tag to the build environment, the `DockerBuildTask` will
apply a
[substitution](https://cloud.google.com/build/docs/configuring-builds/substitute-variable-values)
on the `_IMAGE_NAME` variable. The build configuration must use this tagged name
to build and push the Docker image.

In the case of a custom build using `cloudbuild.yaml`, the `custom_build`
parameter must be set to `True` in the `DockerBuildTask`. An example usage of
`DockerBuildTask` to build the image using the config in
`example_cloudbuild_yaml` follows:

```python
from xlml.apis import task
from dags.vm_resource import DockerImage

docker_image_build = task.DockerBuildTask(
  build_dir="dags/examples/docker/example_cloudbuild_yaml",
  image_name=DockerImage.XPK_JAX_TEST.value,
  custom_build=True,
).run()

# The tagged image name is accessible via `docker_image_build.output`
```
