pluginManagement {
    repositories {
        // Google Maven hosts the real AGP classpath artifact
        // (`com.android.tools.build:gradle`) as well as androidx and
        // com.google.* tooling. No includeGroupByRegex filter here:
        // the filter was excluding the transitive toolchain resolution
        // for some plugin markers in certain environments, and the
        // performance cost of a broader repo is negligible compared to
        // the maintenance cost of a fragile allow-list.
        google()
        mavenCentral()
        gradlePluginPortal()
    }
    // Map plugin IDs directly to their canonical artifact coordinates
    // so Gradle can resolve them from Google Maven / Maven Central
    // without going through the Gradle Plugin Portal's marker
    // redirection. This matters in environments where the Plugin
    // Portal (plugins.gradle.org) is unreachable or where its marker
    // artifacts for a given version have not propagated to the local
    // mirror yet — the symptom is a "could not resolve plugin artifact
    // ...gradle.plugin:<version>" failure even though the underlying
    // tools artifact is sitting right there on Google Maven. The
    // mapping is purely additive: if the marker is reachable, Gradle
    // will use it; if not, eachPlugin kicks in and rewrites the
    // request to the full Maven coordinate.
    resolutionStrategy {
        eachPlugin {
            val id = requested.id.id
            val version = requested.version
            when {
                id == "com.android.application" || id == "com.android.library" ->
                    useModule("com.android.tools.build:gradle:$version")
                id == "org.jetbrains.kotlin.android" ->
                    useModule("org.jetbrains.kotlin:kotlin-gradle-plugin:$version")
                id == "org.jetbrains.kotlin.plugin.serialization" ->
                    useModule("org.jetbrains.kotlin:kotlin-serialization:$version")
            }
        }
    }
}
// No java { toolchain } / kotlin { jvmToolchain } DSL is used anywhere
// in this build — every module pins its Java version through
// `android { compileOptions { sourceCompatibility / targetCompatibility } }`
// instead, and the Gradle daemon JVM is provisioned via
// gradle/gradle-daemon-jvm.properties (foojay URLs are hard-coded
// there, no plugin needed). The foojay-resolver-convention plugin
// was scaffolded originally but is dead code here; re-add it if a
// module later adopts the toolchain DSL.
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "QuickAI"
include(":QuickDotAI")
include(":SampleTestAPP")
