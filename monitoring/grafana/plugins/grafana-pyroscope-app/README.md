<!-- This README file is the one that is displayed on grafana.com website and inside Grafana instances -->

# Grafana Profiles Drilldown

Grafana Profiles Drilldown is a native Grafana application designed to integrate seamlessly with Pyroscope, the open-source continuous profiling platform, providing a smooth, query-less experience for browsing and analyzing profiling data.

Key features include:

- **Native integration with Pyroscope**: Seamlessly integrates with Pyroscope backend to provide detailed performance profiling and analysis.
- **Query-less browsing**: Navigate profiling data without the need for complex queries.
- **AI-Powered flame graph analysis**: uses a large-language model (LLM) to assist with flame graph data interpretation so you can identify bottlenecks, and get to the bottom of performance issues faster.

![Grafana Profiles Drilldown main screen](https://grafana.com/media/docs/explore-profiles/explore-profiles-homescreen-latest.png)

## Before you begin

To use Grafana Profiles Drilldown with Grafana Cloud, you need:

- A Grafana Cloud account
- A Grafana stack in Grafana Cloud with a configured [Pyroscope data source](https://grafana.com/docs/grafana-cloud/connect-externally-hosted/data-sources/pyroscope/) receiving profiling data

To use Grafana Profiles Drilldown with self-hosted Grafana open source or Grafana Enterprise, you need:

- Your own Grafana instance running 11.0 or newer
- Pyroscope 1.7 or newer
- A configured [Pyroscope data source](https://grafana.com/docs/grafana/latest/datasources/pyroscope/) receiving profiling data

## Getting started

Refer to the [Grafana Profiles Drilldown](https://grafana.com/docs/grafana-cloud/visualizations/simplified-exploration/profiles/) documentation.
For instructions installing, refer to the [access and installation instructions](https://grafana.com/docs/grafana-cloud/visualizations/simplified-exploration/profiles/).

## Resources

- [Documentation](https://grafana.com/docs/grafana-cloud/visualizations/simplified-exploration/profiles/)
- [CHANGELOG](https://github.com/grafana/profiles-drilldown/releases)
- [GITHUB](https://github.com/grafana/profiles-drilldown/)

## Contributing

We love accepting contributions!
If your change is minor, please feel free submit
a [pull request](https://github.com/grafana/profiles-drilldown/pull/new)
If your change is larger, or adds a feature, please file an issue beforehand so
that we can discuss the change. You're welcome to file an implementation pull
request immediately as well, although we generally lean towards discussing the
change and then reviewing the implementation separately.

For more information, refer to [Contributing to Grafana Profiles Drilldown](https://github.com/grafana/profiles-drilldown/blob/main/docs/CONTRIBUTING.md)

### Bugs

If your issue is a bug, please open one [here](https://github.com/grafana/profiles-drilldown/issues/new).

### Changes

We do not have a formal proposal process for changes or feature requests. If you have a change you would like to see in
Grafana Profiles Drilldown, please [file an issue](https://github.com/grafana/profiles-drilldown/issues/new) with the necessary details.
