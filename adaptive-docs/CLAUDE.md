# Adaptive Docs - Documentation Site

## Overview

The adaptive-docs service provides comprehensive public-facing documentation for the Adaptive LLM infrastructure. Built with Mintlify, it offers interactive API documentation, integration guides, feature explanations, and troubleshooting resources for developers using the Adaptive platform.

## Key Features

- **Outcome-Driven Content**: Developer experience optimized with immediate value propositions and cost savings examples
- **Progressive Disclosure**: Essential information first, advanced features in expandable sections  
- **Interactive API Documentation**: OpenAPI-powered API reference with live testing and copy-paste examples
- **5-Minute Success Path**: Time-boxed quickstart guide with step-by-step cost savings demonstrations
- **Integration Guides**: Framework-specific guides (OpenAI SDK, Vercel AI SDK, LangChain) with working examples
- **Visual Cost Comparisons**: Before/after pricing examples showing 60-80% savings throughout documentation
- **Actionable Troubleshooting**: Instant-fix guides with working code examples for common errors
- **Responsive Design**: Mobile-friendly documentation with search functionality

## Technology Stack

- **Documentation Framework**: Mintlify for modern documentation sites
- **OpenAPI Integration**: Interactive API explorer with live testing
- **Content Format**: MDX (Markdown + JSX) for rich, interactive content
- **Deployment**: Mintlify hosting with custom domain support
- **Version Control**: Git-based content management with automatic deployment

## Project Structure

```
adaptive-docs/
├── index.mdx                        # Homepage and overview
├── quickstart.mdx                   # Quick start guide
├── authentication.mdx               # Authentication documentation
├── troubleshooting.mdx             # Troubleshooting guide
├── integrations/                   # Integration guides
│   ├── openai-sdk.mdx             # OpenAI SDK integration
│   ├── vercel-ai-sdk.mdx          # Vercel AI SDK integration
│   └── langchain.mdx              # LangChain integration
├── features/                      # Feature documentation
│   ├── intelligent-routing.mdx    # Smart routing explanation
│   ├── prompt-response-cache.mdx  # Caching documentation
│   ├── semantic-cache.mdx         # Semantic caching features
│   ├── provider-resiliency.mdx    # Fallback and resilience
│   └── performance.mdx            # Performance characteristics
├── api-reference/                 # API documentation
│   ├── introduction.mdx           # API introduction
│   ├── chat-completions.mdx       # Chat completions endpoint
│   └── openapi.json              # OpenAPI specification
├── examples/                      # Code examples and tutorials
│   ├── basic-chat.mdx            # Basic chat implementation
│   └── streaming-chat.mdx        # Streaming chat example
├── images/                       # Documentation images and assets
├── logo/                         # Brand assets and logos
├── docs.json                     # Mintlify configuration
├── favicon.ico                   # Site favicon
├── llms.txt                      # LLM-readable site summary
└── README.md                     # Documentation site README
```

## Content Organization

### Navigation Structure

**Documentation Dropdown**
- **Getting Started**: Overview, quickstart, authentication
- **Integration Guides**: SDK and framework integration tutorials
- **Key Features**: Detailed feature explanations and benefits

**API Reference Dropdown**
- **Overview**: API introduction and concepts
- **OpenAPI**: Interactive API explorer with live testing

**Examples & Use Cases**
- **Basic Examples**: Simple implementation examples
- **Advanced Use Cases**: Complex scenarios and best practices

**Support**
- **Troubleshooting**: Common issues and solutions
- **Contact Information**: Support channels and resources

### Content Types

**Getting Started Content**
- Platform overview and value proposition
- Quick start guide with working examples
- Authentication setup and API key management
- Basic configuration and first API call

**Integration Guides**
- OpenAI SDK drop-in replacement instructions
- Vercel AI SDK integration with streaming
- LangChain provider configuration
- Custom integration patterns and best practices

**Feature Documentation**
- Intelligent routing algorithms and benefits
- Caching strategies (prompt-response, semantic)
- Provider resiliency and fallback mechanisms
- Performance characteristics and optimization

**API Reference**
- OpenAI-compatible endpoint documentation
- Request/response schemas and examples
- Error codes and handling
- Rate limiting and usage guidelines

## Configuration

### Mintlify Configuration
**File**: `docs.json`

```json
{
  "$schema": "https://mintlify.com/docs.json",
  "theme": "mint",
  "name": "Adaptive",
  "description": "AI-powered platform for cost-optimized LLM routing",
  "colors": {
    "primary": "#d4af37",
    "light": "#f0e68c", 
    "dark": "#b8860b"
  },
  "api": {
    "baseUrl": "https://llmadaptive.uk/api/v1",
    "auth": {
      "method": "bearer"
    },
    "playground": {
      "display": "interactive"
    }
  }
}
```

### Brand Configuration
- **Primary Color**: Gold (#d4af37) for brand consistency
- **Logo Assets**: Light and dark mode variants
- **Favicon**: Custom favicon for brand recognition
- **Typography**: Consistent with main application design

### API Integration
- **Base URL**: Production API endpoint configuration
- **Authentication**: Bearer token authentication setup
- **Interactive Playground**: Live API testing environment
- **Code Examples**: Multi-language code generation

## Content Management

### Writing Guidelines

**Documentation Standards**
- Clear, concise language accessible to developers of all levels
- Code examples that work out-of-the-box
- Consistent formatting and structure across all pages
- Regular updates to reflect API changes and new features

**MDX Best Practices**
- Use interactive components for better user experience
- Include code blocks with syntax highlighting
- Add callouts and tips for important information
- Embed interactive API examples where applicable

**Code Examples**
- Provide examples in cURL, Python, and JavaScript
- Include complete, runnable examples
- Show both success and error handling scenarios
- Keep examples up-to-date with latest API versions

### Content Maintenance

**Regular Updates**
- Review and update content monthly
- Verify all code examples work with current API
- Update integration guides for new SDK versions
- Add new features and capabilities as they're released

**Quality Assurance**
- Test all code examples before publishing
- Verify links and references are current
- Review for grammar, spelling, and clarity
- Ensure consistent tone and style across content

## Development Workflow

### Local Development
```bash
# Install Mintlify CLI
npm i -g mintlify

# Start development server
mintlify dev

# Preview on localhost:3000
open http://localhost:3000
```

### Content Creation
```bash
# Create new documentation page
touch new-feature.mdx

# Add page to navigation in docs.json
# Edit docs.json to include new page

# Preview changes
mintlify dev
```

### Content Validation
```bash
# Check for broken links
mintlify broken-links

# Validate documentation structure
mintlify validate

# Check OpenAPI specification
npx swagger-codegen validate -i api-reference/openapi.json
```

### Deployment
```bash
# Automatic deployment on push to main branch
git push origin main

# Manual deployment trigger
mintlify deploy

# Check deployment status
mintlify status
```

## Integration with Services

### API Documentation Sync

**OpenAPI Specification**
- Generated from Go backend API definitions
- Automatically updated with API changes
- Includes request/response schemas and examples
- Validates against actual API implementation

**Code Examples**
- Generated from actual API calls
- Tested against development and staging environments
- Updated automatically with API changes
- Include authentication and error handling

### Content Automation

**Automated Updates**
- API reference updated from OpenAPI spec
- Code examples validated against live API
- Feature documentation synced with release notes
- Troubleshooting guides updated with common issues

**Quality Assurance**
- Automated link checking and validation
- Code example testing in CI/CD pipeline
- Content review process for major changes
- User feedback integration and response

## SEO and Discoverability

### Search Optimization
- **Meta Tags**: Optimized titles and descriptions
- **Keywords**: Relevant technical keywords and phrases
- **Structured Data**: Schema markup for better search results
- **Site Speed**: Fast loading times with optimized assets

### User Experience
- **Search Functionality**: Built-in search across all content
- **Navigation**: Intuitive navigation and page organization
- **Mobile Responsive**: Optimized for mobile and tablet devices
- **Accessibility**: WCAG compliance for screen readers and accessibility

### Analytics and Feedback
- **Usage Analytics**: Track page views and user behavior
- **Search Analytics**: Monitor search queries and results
- **User Feedback**: Collect feedback on documentation quality
- **Continuous Improvement**: Regular updates based on usage data

## Troubleshooting

### Common Issues

**Local development issues**
- Install Mintlify CLI globally: `npm i -g mintlify`
- Check Node.js version compatibility (14+)
- Clear npm cache if installation fails: `npm cache clean --force`

**Content rendering problems**
- Validate MDX syntax and formatting
- Check for missing images or assets
- Verify navigation configuration in docs.json
- Test OpenAPI specification validity

**Deployment failures**
- Check Git repository permissions and access
- Verify docs.json configuration syntax
- Ensure all referenced files exist
- Review deployment logs for specific errors

**API documentation sync issues**
- Validate OpenAPI specification format
- Check API endpoint accessibility
- Verify authentication configuration
- Test API examples manually

### Debug Commands
```bash
# Validate documentation configuration
mintlify validate

# Check for broken links
mintlify broken-links

# Test OpenAPI specification
npx swagger-codegen validate -i api-reference/openapi.json

# Preview locally with debug info
DEBUG=1 mintlify dev

# Check deployment status
mintlify status
```

## Performance and Optimization

### Site Performance
- **Fast Loading**: Optimized assets and lazy loading
- **CDN Delivery**: Global content delivery network
- **Caching**: Aggressive caching for static content
- **Compression**: Gzip and Brotli compression enabled

### Content Optimization
- **Image Optimization**: Compressed images with appropriate formats
- **Code Highlighting**: Efficient syntax highlighting
- **Search Indexing**: Optimized search index for fast results
- **Lazy Loading**: Load content as needed for better performance

## Contributing

### Content Guidelines
- **Accuracy**: All information must be accurate and up-to-date
- **Clarity**: Write for developers with varying experience levels
- **Completeness**: Include comprehensive examples and explanations
- **Consistency**: Follow established style and formatting guidelines

### Review Process
- **Content Review**: Technical accuracy and clarity review
- **Editorial Review**: Grammar, style, and consistency check
- **Testing**: Verify all code examples and instructions work
- **User Testing**: Get feedback from actual users when possible

### Documentation Updates
**IMPORTANT**: When making changes to this service, always update documentation:

1. **Update this CLAUDE.md** when:
   - Adding new documentation sections or pages
   - Modifying the site structure or navigation
   - Changing deployment processes or configuration
   - Adding new content types or templates
   - Updating Mintlify configuration or branding

2. **Update documentation content** when:
   - API changes require documentation updates
   - New features are released that need explanation
   - Integration guides need updates for new SDK versions
   - Troubleshooting information needs additions or corrections

3. **Update root CLAUDE.md** when:
   - The documentation site's role in the architecture changes
   - New documentation capabilities are added
   - Integration with other services changes

### Pull Request Process
1. Create feature branch from `main`
2. Make content changes and additions
3. Test locally with `mintlify dev`
4. **Update relevant documentation** (this CLAUDE.md, content pages)
5. Submit PR with clear description of changes and rationale
6. Include screenshots for visual changes
7. Ensure all validation checks pass