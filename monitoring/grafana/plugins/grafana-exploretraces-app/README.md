<!-- This README file is going to be the one displayed on the Grafana.com website for your plugin. Uncomment and replace the content here before publishing.

Remove any remaining comments before publishing as these may be displayed on Grafana.com -->

# Traces Drilldown

Distributed traces provide a way to monitor applications by tracking requests across services.
Traces record the details of a request to help understand why an issue is or was happening.
Tracing is best used for analyzing the performance of your system, identifying bottlenecks, monitoring latency, and providing a complete picture of how requests are processed.

Traces Drilldown helps you make sense of your tracing data so you can automatically visualize insights from your Tempo traces data.
Using the app, you can:

* Use Rate, Errors, and Duration (RED) metrics derived from traces to investigate issues
* Uncover related issues and monitor changes over time
* Browse automatic visualizations of your data based on its characteristics
* Do all of this without writing TraceQL queries

![Root cause latency using Duration metrics](https://grafana.com/media/docs/explore-traces/explore-traces-rate-comparison.png)

## Before you begin

To use Traces Drilldown with Grafana Cloud, you need:

- A Grafana Cloud account
- A Grafana stack in Grafana Cloud with a configured [Tempo data source](https://grafana.com/docs/grafana-cloud/connect-externally-hosted/data-sources/tempo/configure-tempo-data-source/) receiving tracing data

To use Traces Drilldown with self-hosted Grafana open source or Grafana Enterprise, you need:

- Your own Grafana instance running 11.3 or newer
- Tempo 2.6 or newer
- A configured [Tempo data source](https://grafana.com/docs/grafana/latest/datasources/tempo/configure-tempo-data-source/) receiving tracing data

## Getting started

Refer to the [Traces Drilldown](https://grafana.com/docs/grafana-cloud/visualizations/simplified-exploration/traces/) documentation.
For instructions installing, refer to the [access and installation instructions](https://grafana.com/docs/grafana-cloud/visualizations/simplified-exploration/traces/).

## Resources

- [Documentation](https://grafana.com/docs/grafana-cloud/visualizations/simplified-exploration/traces/)
- [CHANGELOG](https://github.com/grafana/traces-drilldown/releases)
- [GITHUB](https://github.com/grafana/traces-drilldown/)

## Contributing

We love accepting contributions!
If your change is minor, please feel free submit
a [pull request](https://help.github.com/articles/about-pull-requests/).
If your change is larger, or adds a feature, please file an issue beforehand so
that we can discuss the change. You're welcome to file an implementation pull
request immediately as well, although we generally lean towards discussing the
change and then reviewing the implementation separately.

### Bugs

If your issue is a bug, please open one [here](https://github.com/grafana/traces-drilldown/issues/new).

### Changes

We do not have a formal proposal process for changes or feature requests. If you have a change you would like to see in
Traces Drilldown, please [file an issue](https://github.com/grafana/traces-drilldown/issues/new) with the necessary details.


<!-- To help maximize the impact of your README and improve usability for users, we propose the following loose structure:

**BEFORE YOU BEGIN**
- Ensure all links are absolute URLs so that they will work when the README is displayed within Grafana and Grafana.com
- Be inspired ✨
  - [grafana-polystat-panel](https://github.com/grafana/grafana-polystat-panel)
  - [volkovlabs-variable-panel](https://github.com/volkovlabs/volkovlabs-variable-panel)

**ADD SOME BADGES**

Badges convey useful information at a glance for users whether in the Catalog or viewing the source code. You can use the generator on [Shields.io](https://shields.io/badges/dynamic-json-badge) together with the Grafana.com API
to create dynamic badges that update automatically when you publish a new version to the marketplace.

- For the logo field use 'grafana'.
- Examples (label: query)
  - Downloads: $.downloads
  - Catalog Version: $.version
  - Grafana Dependency: $.grafanaDependency
  - Signature Type: $.versionSignatureType

Full example: ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?logo=grafana&query=$.version&url=https://grafana.com/api/plugins/grafana-polystat-panel&label=Marketplace&prefix=v&color=F47A20)

Consider other [badges](https://shields.io/badges) as you feel appropriate for your project.

## Overview / Introduction
Provide one or more paragraphs as an introduction to your plugin to help users understand why they should use it.

Consider including screenshots:
- in [plugin.json](https://grafana.com/developers/plugin-tools/reference-plugin-json#info) include them as relative links.
- in the README ensure they are absolute URLs.

## Requirements
List any requirements or dependencies they may need to run the plugin.

## Getting Started
Provide a quick start on how to configure and use the plugin.

## Documentation
If your project has dedicated documentation available for users, provide links here. For help in following Grafana's style recommendations for technical documentation, refer to our [Writer's Toolkit](https://grafana.com/docs/writers-toolkit/).

## Contributing
Do you want folks to contribute to the plugin or provide feedback through specific means? If so, tell them how!
-->
