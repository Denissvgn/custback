#!/usr/bin/env node
/** Non-authorizing installed-artifact acceptance runner. */

'use strict';

const release = require('./verify-release');

function main() {
  try {
    const version = release.verifyPackageSmoke();
    console.log(
      `[custback package smoke] installed artifacts verified for ${version} (diagnostic only)`,
    );
    return 0;
  } catch (err) {
    console.error(`[custback package smoke] ${err.message}`);
    return 1;
  }
}

module.exports = { main };

if (require.main === module) process.exit(main());
