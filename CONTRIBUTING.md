#  Contributing to AI-Driven Lane Detection

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

##  Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [Development Setup](#-development-setup)
- [Contributing Guidelines](#-contributing-guidelines)
- [Code Style](#-code-style)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Pull Request Process](#-pull-request-process)
- [Issue Reporting](#-issue-reporting)

##  Code of Conduct

This project follows a code of conduct that we expect all contributors to follow:

- Be respectful and inclusive
- Use welcoming and inclusive language
- Be constructive in feedback and discussions
- Focus on what is best for the community
- Show empathy towards other community members

##  Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of computer vision and machine learning

### Development Setup

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/lane-tunnel-rt.git
   cd lane-tunnel-rt
   ```

2. **Create Development Environment**
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install development dependencies
   pip install pytest pytest-cov black flake8 mypy
   ```

3. **Verify Setup**
   ```bash
   # Run tests to ensure everything works
   python3 run_tests.py
   
   # Test basic functionality
   python3 main.py examples/sample_video.mp4 --method ransac
   ```

##  Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

-  **Bug Fixes**: Fix issues and improve stability
-  **New Features**: Add new detection methods or enhancements
-  **Documentation**: Improve docs, examples, and tutorials
-  **Tests**: Add or improve test coverage
-  **Performance**: Optimize existing code
-  **Tools**: Improve development and deployment tools

### Contribution Process

1. **Check Existing Issues**
   - Look for existing issues or discussions
   - Comment on issues you'd like to work on
   - Ask questions if something is unclear

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number
   ```

3. **Make Changes**
   - Write clean, well-documented code
   - Follow the code style guidelines
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   # Run all tests
   python3 run_tests.py
   
   # Run specific tests
   python3 run_tests.py --test your_test
   
   # Run with coverage
   pytest --cov=src tests/
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

6. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

##  Code Style

### Python Style Guidelines

We follow PEP 8 with some project-specific conventions:

```python
# Good example
def detect_lanes_ransac(
    frame: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    roi_top_ratio: float = 0.55
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Detect lanes using RANSAC algorithm.
    
    Args:
        frame: Input image frame
        canny_low: Lower Canny threshold
        canny_high: Upper Canny threshold
        roi_top_ratio: Region of interest top ratio
        
    Returns:
        Tuple of (overlay_image, detected_lanes)
    """
    # Implementation here
    pass
```

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Type hints**: Always use type hints for function signatures

### Code Formatting

We use `black` for code formatting:

```bash
# Format code
black src/ tests/ scripts/

# Check formatting
black --check src/ tests/ scripts/
```

### Linting

We use `flake8` for linting:

```bash
# Run linter
flake8 src/ tests/ scripts/

# Run with specific configuration
flake8 --max-line-length=88 --extend-ignore=E203 src/
```

##  Testing

### Test Structure

```
tests/
 test_basic.py              # Basic functionality tests
 test_postprocess.py        # Post-processing tests
 test_fps_tracker.py        # FPS tracker tests
 test_integration.py        # Integration tests (if needed)
```

### Writing Tests

```python
import unittest
import numpy as np
from src.utils.ransac_lane_detection import RANSACLaneDetector

class TestRANSACDetection(unittest.TestCase):
    """Test RANSAC lane detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = RANSACLaneDetector()
        self.test_image = self._create_test_image()
    
    def test_lane_detection(self):
        """Test basic lane detection."""
        lanes, overlay = self.detector.detect_lanes(self.test_image)
        
        self.assertIsInstance(lanes, list)
        self.assertIsInstance(overlay, np.ndarray)
        self.assertGreater(len(lanes), 0)
    
    def _create_test_image(self):
        """Create test image with lane features."""
        # Implementation here
        pass
```

### Running Tests

```bash
# Run all tests
python3 run_tests.py

# Run specific test file
python3 -m pytest tests/test_postprocess.py

# Run with coverage
python3 -m pytest --cov=src tests/

# Run specific test
python3 -m pytest tests/test_postprocess.py::TestPostProcessing::test_lane_detection
```

### Test Requirements

- All new features must have tests
- Tests should be fast and reliable
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies when appropriate

##  Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Follow Google docstring format
- Include type hints
- Provide usage examples for complex functions

```python
def preprocess_image(
    image: np.ndarray, 
    method: str = "clahe_gamma"
) -> np.ndarray:
    """
    Preprocess image for lane detection.
    
    Args:
        image: Input image as numpy array
        method: Preprocessing method ('clahe', 'gamma', 'clahe_gamma', 'hist_eq')
        
    Returns:
        Preprocessed image as numpy array
        
    Example:
        >>> import cv2
        >>> image = cv2.imread('test.jpg')
        >>> processed = preprocess_image(image, 'clahe_gamma')
    """
    # Implementation here
```

### README Updates

When adding new features:
- Update the features list
- Add usage examples
- Update performance benchmarks if applicable
- Update project structure if new files are added

### API Documentation

- Document all public APIs
- Include parameter descriptions
- Provide return value descriptions
- Add usage examples

##  Pull Request Process

### Before Submitting

1. **Ensure Tests Pass**
   ```bash
   python3 run_tests.py
   ```

2. **Check Code Style**
   ```bash
   black --check src/ tests/ scripts/
   flake8 src/ tests/ scripts/
   ```

3. **Update Documentation**
   - Update README if needed
   - Add docstrings for new functions
   - Update type hints

4. **Test Your Changes**
   ```bash
   # Test with different methods
   python3 main.py examples/sample_video.mp4 --method opencv
   python3 main.py examples/sample_video.mp4 --method ransac
   
   # Test with different preprocessing
   python3 main.py examples/sample_video.mp4 --preprocessing clahe_gamma
   ```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Test addition/improvement

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented if necessary)
```

### Review Process

1. **Automated Checks**
   - Tests must pass
   - Code style must be correct
   - No merge conflicts

2. **Manual Review**
   - Code quality and correctness
   - Test coverage
   - Documentation completeness
   - Performance impact

3. **Approval**
   - At least one approval required
   - All requested changes addressed
   - CI/CD checks pass

##  Issue Reporting

### Before Creating an Issue

1. Check existing issues
2. Search closed issues
3. Check documentation
4. Try latest version

### Issue Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.8.10]
- OpenCV: [e.g., 4.8.0]
- PyTorch: [e.g., 2.0.0]

## Additional Context
Any other relevant information
```

### Feature Requests

```markdown
## Feature Description
Clear description of the feature

## Use Case
Why is this feature needed?

## Proposed Solution
How should this be implemented?

## Alternatives
Other solutions considered

## Additional Context
Any other relevant information
```

##  Release Process

### Version Numbering

We use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number updated
- [ ] CHANGELOG.md updated
- [ ] Release notes prepared
- [ ] Tag created

##  Getting Help

- **Documentation**: Check README.md and PROJECT_STATUS.md
- **Issues**: Search existing issues or create new one
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Ask for help in pull requests

##  Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to this project! 
