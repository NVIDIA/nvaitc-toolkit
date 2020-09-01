# Contributing to NVAITC Toolkit

Please read this document to understand how to contribute to this project:
 * [Report an Issue](#report-an-issue)
 * [Contribute Code](#contribute-code)

## Report an Issue

If you find a bug you are welcome to report it.
We can only handle well-reported, actual bugs, so please follow the guidelines below.

Once you have familiarized with the guidelines, you can go to the [GitHub issue tracker for NVAITC Toolkit](ADD LINK) to report the issue.

### Quick Checklist for Bug Reports

Issue report checklist:
 * Real, current bug
 * No duplicate
 * Reproducible
 * Good summary
 * Well-documented
 * Minimal example
 * Use the [template](ISSUE_TEMPLATE.md)


### Issue handling process

When an issue is reported, a committer will look at it and either confirm it as a real issue, close it if it is not an issue, or ask for more details.

An issue that is about a real bug is closed as soon as the fix is committed.


### Usage of Labels

GitHub offers labels to categorize issues. We defined the following labels so far:

Labels for issue categories:
 * bug: this issue is a bug in the code
 * feature: this issue is a request for a new functionality or an enhancement request
 * design: this issue relates to the UI or UX design of the tool

Status of open issues:
 * help wanted: the feature request is approved and you are invited to contribute

Status/resolution of closed issues:
 * wontfix: while acknowledged to be an issue, a fix cannot or will not be provided

The labels can only be set and modified by committers.


### Issue Reporting Disclaimer

We want to improve the quality of CLA Assistant and good bug reports are welcome! But our capacity is limited, thus we reserve the right to close or to not process insufficient bug reports in favor of those which are very cleanly documented and easy to reproduce. Even though we would like to solve each well-documented issue, there is always the chance that it will not happen - remember: CLA Assistant is Open Source and comes without warranty.

Bug report analysis support is very welcome! (e.g. pre-analysis or proposing solutions)


## Contribute Code

You are welcome to contribute code to NVAITC toolkit in order to fix bugs or to implement new features.

There are three important things to know:

1.  You must be aware of the Apache License (which describes contributions) and **agree to the Contributors License Agreement**. This is common practice in all major Open Source projects.
 For company contributors special rules apply. See the respective section below for details.
2.  There are **several requirements regarding code style, quality, and product standards** which need to be met (we also have to follow them). The respective section below gives more details on the coding guidelines.
3.  **Not all proposed contributions can be accepted**. Some features may e.g. just fit a third-party add-on better. The code must fit the overall direction of CLA Assistant and really improve it. The more effort you invest, the better you should clarify in advance whether the contribution fits: the best way would be to just open an issue to discuss the feature you plan to implement (make it clear you intend to contribute).

### Contributor License Agreement

When you contribute (code, documentation, or anything else), you have to be aware that your contribution is covered by the same [MIT License](https://opensource.org/licenses/MIT) that is applied to NVAITC Toolkit itself.
In particular you need to agree to the Individual Contributor License Agreement,
which can be [found here](nvaitc-toolkit-cla.md).


### How to contribute - the Process

1.  Make sure the change would be welcome (e.g. a bugfix or a useful feature); best do so by proposing it in a GitHub issue
2.  Create a branch forking the cla-assistant repository and do your change
3.  Commit and push your changes on that branch
4.  In the commit message
 - Describe the problem you fix with this change.
 - Describe the effect that this change has from a user's point of view. App crashes and lockups are pretty convincing for example, but not all bugs are that obvious and should be mentioned in the text.
 - Describe the technical details of what you changed. It is important to describe the change in a most understandable way so the reviewer is able to verify that the code is behaving as you intend it to.
5.  If your change fixes an issue reported at GitHub, add the following line to the commit message:
    - ```Fixes #(issueNumber)```
    - Do NOT add a colon after "Fixes" - this prevents automatic closing.
6.  Create a Pull Request
7.  Follow the link posted by the CLA assistant to your pull request and accept it, as described in detail above.
8.  Wait for our code review and approval, possibly enhancing your change on request
    -   Note that the CLA Assistant developers also have their regular duties, so depending on the required effort for reviewing, testing and clarification this may take a while

9.  Once the change has been approved we will inform you in a comment
10.  We will close the pull request, feel free to delete the now obsolete branch
