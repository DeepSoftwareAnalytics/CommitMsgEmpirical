merge pull request from kochizufan clickexperia
fix snapgrid bug
fix no clicks in no swiping areas
move stopautoplay into destroy method
send swiper instance to render
scrollbar fix pointer position calc
fully respect the law of demeter
use innerwidth instead because of jquery bug
fix file names
isomorphic layout effect
merge pull request from jsundn master
remove unused parameter
resolve bug restricting translateto to specific bounds
fix loop mode with fractional slidesperview setting
fix loop behavior
fix none working input in jquery version
use ssr window document
better iphone detection
clean package before release
add threshold param for allowed time between wheel scrolls
release locks during resize
remove old parameter
merge pull request from vaxul master
fix zoom for rtl
fix once events unbinding
fix error in non loop mode
doesn t have effect on stoppropagation
fix syntax error
merge pull request from vladimirivanin master
merge pull request from pantcaser
update gulpfile js
add speed parameter
fix a typo in disabledraggable
additional check for pagination
check for initial virtualtranslate
minor update to argument names to better adhere with existing translate documentation and code
add allowslidetoprev and next only enable slideto if true
avoid loading simulation when is picture
sort prepend indexes disable setwrappersize
reverted to older commit
add updateonwindowresize parameter
fix roundlengths for wrapper translate
remove version from code
add centeredslidebounds parameter prevents gaps at beginning and end of slider
merge pull request from knusperpixel master
adds prefix initialized class to the container element
merge pull request from ashot kr round fix
merge pull request from jsundn master
fix issue with not considering padding
remove virtual update from beforeinit
disabled by default
fix method name
merge pull request from ghaiat feat shadow
merge pull request from webmasterlpo master
remove redundant condition
fix issue with wrong row column if not full groups
merge pull request from glebmachine fix clear events
fix eslint error math pow is restricted from being used use the exponentiation operator instead no restricted properties
fix an issue where swipeprev will not work in chrome when zoomed in
add touch start prevent default option
fix direction in transitionend callback
update progress if needed
return false in handlekeyboard if keycode matches locked direction
merge pull request from azaslavsky master
fix rtl scrollbar
modified checkoverflow to account for slide offsets
fix per column row slides fill
fix virtual slides index in prependslide
fix wrong object of params
swapped handler apply after if in class js
fixes issue with building pagination when s params slidesperview is auto and slider should be looped
option to invert mouse wheel
support for multiple pagination containers
fix fix runcallbacksoninit bug
fix win chrome clicks on title attibute links
merge pull request from ironkinoko master
merge pull request from depoulo master
update release js
add breakpoint event
add observeslidechildren parameter
use diff consts
fix mixed operator
import new modules
destroy keyboard mouse
remove package documentation
prevent keyboard control with pressed metakeys
add freemodemomentumvelocityratio parameter
remove empty line
fix parallax still being applied when disabled
don t set same value
bug fix centered slides show hide position
fix in fire callback
merge pull request from ragnaroek master
fix support react virtual rtl direction
optimized updateslides code
check for gestures support
update core js
merge pull request from michaelwoe michaelwoe bugfix thumbnails cannot read property of null
use method instead of accessor
double check for event listeners
use ssr window
fix touchevent remove unused check
remove white space
update hash navigation js
fix row calc
disable accessibility by default
save dom to swiper
merge pull request from ccebrand master
fixes controller linearspline is not a constructor error
better lookup for clicked slide
merge pull request from jasonlav master
merge pull request from exodusanto master
update resize observer events handlers
followfinger respect threshold
move emit later
changed varname in stopautoplay
fix issue with slider not updating the current position properly when swipeto function is called externally
loop when swipe locks
add slide index if not presented
merge pull request from michaelwoe master
resize on observerupdate
swiper destroy fix for multi invocation of detachevents function
don t set grab cursor in case of no simulatetouch
fix tripple equal
disable by default
add centeredslidesbounds defaults
remove blank line in block fix padded blocks eslint error
merge pull request from brunolm feature force prevent default
update classes before ontransitionstart callback
add support for ios in app uiwebview edge swipe detection
sets keyboardcontrol param in disablekeyboardcontrol and enablekeyboardcontrol
merge pull request from kochizufan loopfix
feature thumbs add option to control whether multiple thumbnail slides may get activated or not
fix for mouse wheel free mode loop
don t assign undefined handlers
core change default toucheventstarget container wrapper
fix activeslide reset at init
adds check for navigation object fixes
add label to formelements
don t produce callback in case of swiper destroyed
call virtual update only once when initialslide
mousewheel fix forcetoaxis inverts scrolling
fix incorrect calculation of number of full columns
reset translate after remove all slides
save locks after breakpoint
fix issue where when swiper is imported as a module it throws errors because this is undefined
correct min translate
merge pull request from nova tan bugfix resize causing carousel slide to last item
remove unused styles
fix chrome broken mousewheel
move hash update to transitionend
merge pull request from evidica master
fix init condition for numofmainbullets
firefox can t handle event srcelement add fallback to event target
better slide index order calc in slidespercolumnfill row
fix parallax breaking on breakpoint change
fix listeners leak in autoplay mode
merge pull request from trevld master
better check for short swipes over nav buttons
merge pull request from danielruf patch history destroy
mouseleave handler name bug fix
remove unused modular instance
fix not passing correct this to onswipercreated callback
fix fix runcallbacksoninit bugs
rename allows add init true parameter
remove unused var
window screen polyfill
update breakpoint params
fix last slide issue when initialised with no size in loop mode
better check for allowclick
merge pull request from lcharbon master
add option to disable swiping on noswiping class
check for releaseonedge params on mousewheel
fix wrong hash navigation issues
update setgrabcursor js
ios bug with not firing resize event
fix transformed slides size calc
core rename uiwebview
mark active slide with a configurable class
core fix onresize handler
don t prevent default in case of disabled touchmove
run callbacks on init
emit index change events
fix lint update lint
loop and autoplay bugfix
merge pull request from brightcells master
fix ie blur
test for ie with touch
add an event on hash change
core fix momenum bounce bug on ios
fix js banner
better support for scrollcontainer
merge pull request from barefootchild barefootchild fix runcallbackoninit
add missing import for support module
missed to declare allowslidetoprev and next properties in defaults
fix add slide with active paging
fix null srcset
merge pull request from garbanas bugfix flexbox support
merge pull request from jsundn master
onslidenext prev callbacks
update lazy load js
fix ci issues
rtl for flip
merge pull request from tom sherman target touches safari fix
add component support
remove unused defaults
update autoheight on lazy loaded
merge pull request from pagepro mousewheelcontrol param update
new waitsfortransition parameter
autoplay by slide
add before change slide event
fix handler unsubscription error on multiple create destroy events in a row
try to fix lint
fix noswipingclass functionality
add missing use strict statements
ensure observerupdate event is only dispatched once on a mutationevent
fix previousslideslength var
don t use getboundingclientrect
add virtual component
remove blank slides on loop destroy
allow capital letters in history
don t track all mouse buttons
merge pull request from nukefusion columncalcfix
update idangerous swiper js
fix for safari emulator
remove redundant method call
fix resistance bounce with swipe locks
fix autoplay bound context
merge pull request from danielkalen fix noswiping
fix multiple touch issue
core consider scrollbars sizes
fix fix runcallbacksoninit bug
don t trigger dom events
update unsetgrabcursor js
fix mousewheel fail
vue clone slides
fix freemodesticky in rtl mode
fix virtual slides sort
dom add styles method
add new onobserverupdate callback
merge pull request from nealnote master
rename slidesperview slidesperviewdynamic
lazy load on resize
autoplay unset paused flag after visibility change
remove also data attributes
better cleanup when instance is deleted
new lazyloadingclass parameter
fix swiper onclick don t move in chrome on old android os android lollipop marshmallow
bug fix bug swiper stopautoplay swiper autoplay stop
fixes issue with wrong obj copy
fix slides spread
add before init event
also check for animation before preventing click
preventintercationontransition false by default
don t enable when disabled
fix right click
merge pull request from websirnik master
implement missing destroy method for swiper history
fix order calc in multirow
fix support pointerevents
fix condition for emmiting slidechange event
add next prevall methods
set virtual translate
return duration within onsetwrappertransition callback
fix pagination bullet class mitstake
add enable and disable keyboardcontrol functions to api
merge pull request from anteprimorac fix parallax typo
merge pull request from luis kaufmann silva master
fix bullet reference
build remove console log types
correct append slide for rtl
merge pull request from visualspace master
add pagination fraction formatters
add lock and unlocks functions for slideto
update ontouchmove js
lowercase static props
emit transition events
move back setwrappersize
issue with input field in swiper
fix thumbs swiper with virtual slides
make possible of work shift mouse wheel horizontal scroll
rename scrollbar init to scrollbar set
use changedtouches instead of targettouches
fix flip effect event
add requirejs support
fix locks reset
clicked virtual slide
remove console logs
core virttual fix before after vars
core install modules in core class
fix prevented click event
lint full fix
rtl bug fix
fix issue with autoplay and fade effect
set breakpoints on update method
don t pass none to webkitcssmatrix constructor
unused detach variable
add beforeresize event
fix breakpoint with loop getting wrong realindex when on init
fix usual breakpoints
merge pull request from rayvincent bsd master
update swiper js
remove console log
ci error fix
loop pagers autoplay fix
react fix autoplay param name
merge pull request from pimlie fix cancel touchmove event
swiper destroy fix for multi invocation of detachevents function
disabled by default
add comma lint
update meteor package description
switch to using classname vs classlist for better backwards compatibility
add enabletouchcontrol and disabletouchcontrol
fix only cancel cancelable touchmove events
check also for swiper destroyed
fix parent swiper jump
fix popacity and pscale typos
mousewheelcontrol param setting when updating this using mousewheelcontrol methods
switch mouse target to document
cross fade option for fade effect
merge pull request from andyburke master
merge pull request from daleknauss master
fix callback name
fix preloader selector
disabled by default
fix umd export
correct constructor call
remove data swiper slide index attribute on destroyloop
check for oninit instead of onfirstinit before calling oninit
snap on release by default
replace startswith with indexof
lazy load on scrollbar drag
fix lazy loading in jquery
merge pull request from azaslavsky disable touch control
enable disable mw control
fix issues with loop and mousewheel
merge pull request from zefoy fix destroy
normalize centeredslides slide progress
remove old code
fix path ends with
information about callbacks
fix in pagination detect uniquenavelements
int real index
use correct check for nav button as target
use clickable class instead of modifier with postfix
prevent default for non touch event
fix not working transitionend in fade effect
solution to issue
fix swiping in edge browser
merge pull request from hiroqn master
remove wrong copy paste
add destroy method
fix autoplay in loop mode
prepare for flip effect
remove unused parameter
lazy loading with mousewheel
update eslintrc js
merge pull request from triloworld master
merge pull request from exodusanto master
fix a typo in updatesize
overwrite original params
update build js js
fix ipod device detection
zoom check for cancelable event
fix typings build
merge pull request from jamesryanbell master
emit edge events
return in case of array
add forcepreventdefault flag
fix slidesperview type
fix index in fade effect
update progress on visibility
move to static class fields
fix lazy override
fix dynamic bullets when not enough bullets
merge pull request from voxtex fix lazy loading bug
fix for issue
fix core methods paths
merge pull request from rdsn master
emit keypress event at the end of keyboard control
improve console method stubbing
add new console api methods to console log protection
add blank lines to script js
don t wrap all args from log to console log with array
fix typo in gulpfile js
comprehensive protection against stray console log s thx henrik fixes
document write bugfix from fearphage
fix two bugs introduced in the multiline reformat
updating log in script js
merge pull request from kirbysayshi master
remove empty line from main js
fix license banner in main css after
merge pull request from philipwalton strict log
fix console stubbing method in oldie
merge pull request from nchaves master
make build set permissions in archive correctly
window log enhancement thnx trevnorris fixes
remove unneeded comma from gulpfile js
remove duplicate marktimeline from plugins js
formatting style correction
fix update yarn workspaces error prompt link
fix options explicit conventional prerelease
fix use correct instance property override
make sure lernajson is defined
push version tags explicitly
fix custom version prompt
feat version refresh package manifests after preversion lifecycle
avoid false positives during integration test filtering
fix version pass atomic to git push
fix import better handling of patch is empty
refactor run lifecycle reasonable default for opts unsafe perm
log stdout on run command errors
merge pull request from kittens private
chore mocks add npmpublish order helper
fix npm publish accept opts log defaulting to libnpm log
chore test helpers ensure callsback waits for the next tick
use message passed as argument to the static method input in promptutilities
fix checkupdatedpackages for repos with no tags
pass in tags argument to git describe
use mkdirp instead of fs mkdir to allow for scoped packages
fix add order short flags first in help output clarify description
check if directories exist before remove during hoist
refactor command return result of execute in promise chain
remove a from git tag invocation
test unmock collect updates when testing publish from git
fixes typeerror during lerna bootstrap
fix options explicit force publish
fix collect updates remove period from committish log message
test run lifecycle use package instance in test instead of bare object
chore helpers add format option to getcommitmessage
add support for new yarn workspaces config format
chore flailing around trying avoid no visual difference snapshot garbage on windows
chore add helper methods to write pkg mock
fix exec allow config files to override defaults
fix command detect composed commands more accurately
fix publish use correct field name when limiting retries
merge pull request from kittens remove stable tag
fix publish allow unpublished packages to pass access verification
bin remove yargs terminalwidth workaround
fix options explicit use workspaces
fix spelling mistake in command js
fix npm conf otp should default to undefined figgy pudding is very strict
uncomment git push
fix exec handle node string error codes before setting process exitcode
feat cli set config ci from env var
fix collect updates match dotfiles when ignoring changes
refactor listable pass explicit arguments to batch packages
refactor gitutilities diffsincein expects a committish not name version
escape double quote replace cpr with fse
fix import logger info typo
spawn child process for rimraf
fix init lerna init does not in fact require git
test integration use correct independent tag patterns
chore retrieve package version before directory is changed in command test
durable includefiltereddependencies config via lerna json
chore package graph add inspect helper to workaround jest bug
fix publish checkout manifest changes serially
fix conventional commits ensure potential validationerror in getchangelogconfig is propagated correctly
refactor repository remove unused nodemoduleslocation getter
use close event to wait for spawned processes to finish
chore remove references to jasmine global
add tests for npmutilities splitversion
make pkg argument optional for lerna diff
show help when invalid command
fix npm dist tag port upstream npm cli
chore use multiline template strings for verbose error messages
detect changes in packages with prefix
fix lscommand add list alias
test integration avoid inexplicable snapshot comparison errors on windows
fix add other potential cause of git initialization failure
fix version mismatch in lerna bootstrap
fix spelling of resetting
feat npm publish add npmpack export
test integration always turn of chalk colors to avoid snapshot pollution
fix publish heighten dist tag success log level to distinguish from noisy fetch logging
test fix lint error
fix clear mocks during publish integration test
refactor test assert on loadjsonfile registry keys not snapshot
fix npm publish tip toe around logging when emitting chunk to stdout
fix silly level log output for conventional commits
use logifysync decorator for filesystemutilities readdirsync
feat package graph add rawpackagelist getter
fix create symlink use octal notation for chmod values
don t chain off of stream setencoding
fix import avoid unrecognized input error from colors when using flatten
refactor command do not store raw packages list as instance property
ensure files end with newlines properly
fix collect updates derive most recent tag from annotated tags only
style fix indentation
fix run re order npm client in help output
fix bootstrap early exit condition
fix version resolve prerelease for version without bump
feat run lifecycle resolve target package to aid chaining
fix exec clarify no bail option
refactor publish snapshot options passed to n r f auth helper
fix callback params
test windows pass fix env to all exec integration tests
chore serialize tempdir remove desperate logging
fix do not pass custom preset to recommended bump
fix list restore empty json array output when no results
fix version search for complete tag prefix when composing github releases
test pre cache command module to fix ci
chore add options argument to run lifecycle mock
fix changed clarify early exit log message
remove stable tag
chore remove irrelevant eslint overrides
fixes diff output colors
refactor publish allow git reset to be configured from lerna json
fix tests and add another
warn user and exit non zero if command is invalid
use correct logger method for warnings
refactor command don t bother filtering logged argv
fix use relative path for file dependency
feat filter options allow command to continue if no packages are matched
fix publish check for git describe error explicitly
make unlinksync synchronous fixes
fix update ignore on windows
style typo seperated separated
fix global options allow sort to be configured from file
config flags shouldn t be undefined for publish
refactor encapsulate semver comparison in packagegraphnode
don t prune the repo root when hoisting
fix version prevent clobbering composed yes option
feat global options add hidden ci option
refactor project expose default package glob as static property
feat ls log number of packages listed
fix run allow config files to override defaults
allow additional properties in lerna init
add lerna as a devdep
fix clean stack try to avoid causing errors during error cleanup
fix copypasta use correct import path
fix npm conf do not overwrite defaults with undefined cli keys
fix imports for runcommand and execcommand
fix publish allow tag check to fail with strong warning
just emit usage when run with no command
test config ensure clean install test aka npm cit uses clean jest cache
support sort option in lerna json
fix create use filename without scope when generating imports from test file
feat child process allow exec opts stdio override
fix integration limit concurrency of bootstrap hoist test
fix publish add default description npm for npm client
test mocks reset collect updates registry after each test
let git am to fall back to three way merge during import
lerna init should add lerna as a devdependency
add missing root level version lifecycle calls
chore publish correct coverage ignore
be abel to change the npm dist tag
temp remove add private in package json during init
test properly disable progress bars
test version pass explicit no ci to ensure error is thrown
fix typo bootstraped bootstrapped
fix disable progress in commands that only output lists
refactor version tweak error code when remote branch does not exist
test helpers stop being nice about ci git garbage
fix publish write temporary annotations once not repeatedly
feat npm publish pass non interactive to yarn publish
update generated changelog header to point to conventional commits skip ci
fix options explicit pre dist tag
fix collect updates remove redundant short circuit
feat publish refresh package manifests after prepare prepublishonly lifecycle
chore fix typo in comment
return error and exit on invalid command
test version ensure prompt default preserves prereleaseid
get rid of confusing tostring name
chore use fs extra json methods in helpers updatelernaconfig
chore don t destructure lodash require
test git init set user name email directly during fixture init
chore map to registry ensure env does not pollute config resolution
merge pull request from joscha jpnelson master
add support for root level version lifecycle
grammar the things
fix describe ref fallback refcount is the number of commits since beginning of repository
update independent flag warning to be more descriptive
merge pull request from ianwremmel master
chore helpers disable overzealous lint rule
refactor publish annotategithead should map over packagestopublish
refactor version use package set method to assign package version
test project remove load json file stub
test git utils fix windows garbage
refactor packageutilities use pkg manifestlocation
fix command enable progress from top level log object
chore windows join the wack to the whack path
test pack directory remove irrelevant assertion
fix use localecompare to sort packages by path
fix publish guard against undefined tag
test integration avoid quoting arguments to coddle windows
test windows don t assert on platform dependent env var interpolation
test helpers clones should always have a dang main branch
refactor add map scope names after makechanges
print the updated packages at the end for use in the changelog
test integration reduce flakiness of error message test
use let to allow re assignment of undefined lernajson
refactor validation error pass along remaining arguments to logger
disable progress bar in non tty environments
add diff to meow
add missing exit
test config do not bail on integration tests
fix execsync use
fix publish correct silly log heading
merge pull request from jmm typo
test npm publish use actual package instance in tests
refactor project don t mutate packageconfigs parameter
allow concurrency to be configured via package json
implements add a confirm flag
fix create skip repository property when git remote is missing
short circuit out of install with no packages
use correct logger method in package method
merge pull request from gigabo explicit tag push
fix publish add default for tag version prefix
fix publish get tagged packages from merge commit
feat collect packages simplify signature
chore use exponential notation in integration test timeout setup
chore test add workaround for bad jest caching behavior
fix run lifecycle omit circular options from config
refactor link use package graph getter instead of instance packages
lint remove unnecessary escape character
replace force version with force publish and alias it
refactor project inline array flattener
fix toposort does not need to be a lazy getter
test serialize changelog fix commit url regex
fix version correct warning message
merge pull request from hzoo change npm dist tag
fix init lerna json does not exist in empty repository
test integration ensure checkout is from upstream of clone
chore helpers pass explicit diff prefixes to avoid localized variation in snapshots
fix version log message when git repository validation is skipped
fix git utils pass no verify to git push
test integration assert on stable result of git status porcelain
fix add snapshot opts passed to pacote manifest
syle rename confirm to yes
merge pull request from cab scope support
fix filter options require arguments to scope and ignore
pass color auto to avoid color escapes when piping diff output
test listable turn off chalk so stuff doesn t fail in ci
lint fix default case error in a way prettier doesn t mangle
fix add avoid passing bad scope to pacote
fix options explicit conventional graduate
fix package resolve correct relative spec
refactor publish condense refresh after pack
test helpers workaround ancient git client in ci
test run improve coverage
refactor publish clarify intent of inverted boolean options via instance properties
fix hoist with no argument default
disable all non error logging when stdout is piped
fix run yarn in non interactive mode
refactor gitutilities diffsincein should slash location path
test import run git setup sequentially parallelism messes up windows
fix filter options allow private to be configured from file
fix rimraf bin resolution
indicate private modules in output from updated
flaky test don t expect settimeout to be exact
use npm for auto install by default
fix issue with normalizeoptions and publicurl
fix property descriptor of config internal in transforms babel js
clireporter remove stdout proxy as ink now addresses this
only close the watcher after the ready event is emitted
fix dotenv package error
parse port string as integer
add support for firefox s moz extension protocol to bundle url js
log errors on pluginload
validate if a postcss config is an object
default port to process env port
remove unnecessary invariant
set terser s toplevel flag with commonjs as well
no longer use unref in workers
fix set default pragmafrag option bug
default package to empty object
fix page reload loop with react refresh
don t load existing sourcemaps if sourcemaps are disabled
remove url search and hash
update json reporter
template literal fix
print stack traces on error
pug deprecated pretty
fix postcss config plugin check
fix promisequeue returning null when there are no jobs
node sass accepts importer as single function or array of functions
ignore dynamic sourcemap code
await ending workerfarm in reverse handle test
fix type import
fix devserver flow violations with mime
remove h alias for hmr port option
merge pull request from marcioj marcioj set env
feat resolver give high priority to extension of parent file
fix dependency list parsing in rustasset for paths on windows
don t clear console in the tests
use this write and super end for jspackager
fix unintended vue asset supplemental code insertion
watch realpath and always emit hmr updates
fix autoinstall of cssnano
return early instead of throwing
improve windows emoji support detection
remove empty bundles and bundle groups
refactor remove unnecessary not null check
use buffer from
merge postcss modules config
add loglevel option to bundler
use project relative path in the asset comments
point pcss files at cssasset
don t mirror permissions in browser builds
catch realpath errors
lookup correct generated output for bundle type in rawpackager
hardcode cwd to parcel repo root for parcel babel register
skip register tests for now
fix transformer svgo use asset id as extra path
only use os cpus and filter out core like physical cpu count did
update post process to recive final assets rather than initial ones
remove unnecessary mkdirp
simplify bundled babel loading in transformer
add hmr port and hmr hostname options to parcel watch
clean existing lints and enable recommended flowtype lints
don t modify script nodes with text html type
put back name in mappings
linecounter with empty string
update runtimes to reflect removals aren t references anymore
fix use require resolve when resolving default config
fixes make help same as help command
fix hang up when a lot of parallel operation request the file system
move jsasset s options gathering code into separate method
re enable cli reporter
ensure that parent bundles are invalidated when inline bundles change
try another way of raising the timeout
refactor json stringify replacer can not be boolean
don t need weakmap for properties of same type
clear console before accepting updates not after
import node directly from types package
fix dev server index file
set ora discardstdin option to false
revert use modern browsers in all development
normalize filenames to nfc for hfs
terser also mangle toplevel with commonjs
forward extra properties from pug config to pug
deterministic bundle trees
atomic writes rename on close not finish
fix vscode syntax highlight with postcss in vue component style tag
check if there is an original mapping before trying to get location of that mapping
default to ending workerfarm
react fast refresh devdependency
remove statement from experimenting
fix indented syntax type for single file vue components in assets sassasset js
fix comment typo
fix ora spinner in ci environments
programatically pass env vars as a whitelist
fix accessing the global object
add charset to error response
always add bundle loader module to the bundle
patch hmr use hostname for ws instantiation
waise timeout of watcher test
fix error when node id is undefined instead of null
fix css hmr by appending payload as a new style element
disables stdin raw mode on exit
enable recommended react eslint rules across repo
merge pull request from fansenze master
clear the console in browser on each hmr
don t swallow resolver error
cli option for setting public path to use in serve mode
bug fix make sure bundles is an array in loadbundleslazy
strict mode prelude
don t match absolute paths for url dependencies
don t start server if target isn t browser
pass json support tests
invalidate asset requests when included files change
remove call to window location fix
use absolute path in parcel babel register execarg
generate unique certificate serial numbers
pass bundler to packagers
remove mismatching assets from bundle assetgraph
connect inbound bundles to bundlegroups when they re created
don t resolve slash and tilde paths twice
fix fs test for mac
remove arrow function syntax in hmr runtime
set integration test loglevel to none
allow dotfiles to be served
add hot module replacement port
add deserializer function to config
fix imports in parcel bundler
disable flakey macos fs events in watcher in the tests
events fix tests for alreadydisposederror
reload css from same origin
check asset membership before remove from graph
use path sep instead of hardcoded forward slash
add jsx to resolver extensions
restrict deep imports
fix build not exiting in dev env
add missing no autoinstall flag to build command
delay dispatching of buildsuccess buildfailure events
treat svg to react transform as jsx instead of js
send cors headers when a file does not exist
remove extra argument passed to addassettobundle in jspackager
bundle name hash key generation is not environment independent
make reporterrunner opts exact
https text fix
fix require path in react refresh transformer
add a builderror event to bundler
make ui state exact
wait for the next event loop iteration when loading wasm
don t package an empty bundle
don t transform json
wait for file stream to finish before finishing bundle
fail early on plugin load and throw clear errors
add bundle id to bundle hash
use the test to assert this child killed rather than checking time difference
make sure handle is registered with the serializer
accept a number in cli reporter gettext
remove circular require in asset js
add an bundlestart event
cross platform deterministic asset ids
disregard order in dev deps when creating cache keys
fix bug that continues to package optimize even when cache entry is found
merge pull request from robertwhurst fix post process assets
fix failing appveyor test
count size past the buffer limit
import inspector only if it s available
allow https for watch
fix posthtml breaking svgs
add a warning emoji to the logger
fix loadplugin when the plugin have some async work while loading
be sure to run requests that have not produced a valid result
fix writing files in subfolders inside dist directory
update lib src in eslint plugin tests
ensure bundlereference is registered with the serializer
fix remove date tolocalestring in logfile names to prevent slashes
fix security vuln
add return types in promisequeue
mention babel minify
don t cache dynamic bundles that had a network error
throw when dist target does not match transformed entry
add data so it is possible to include globals scss variables
standardize comments on empty implementation method
avoid crash when max old space size is present
replaced fwd slashes with backslashes for win to fix sass deps watch
fix error when no targets options in babelrc
mark packages as public
restore defaultprops to spinner
do not use window global in hmr runtime
always parse dependencies if babel ran fixes
don t default scopehoist option to true
add server reference to bundler
merge typescript configuration with defaults if found
add validation for empty parcelrc
fix sass dependencies can not be watched when includepaths is a relative path
fix vue asset supplemental code concatenation
fix improve error message for urls without pathname
fix bundle url matching
include pipeline in cache key
fix sourcemap file size in report
respect log levels in cli reporter
allow no scope hoist as a cli option for builds
expose asset and packager
disable babelrc when building only modules syntax
iterate nodes to copy instead of traversing
cancel previous build at start of new build
server clear error on success
ignore cache and git folders from watcher
remove requestgraph addsubrequest
use bundled when asset is not source
mark bus esm todo
merge pull request from yeskunall typo in asset
catch errors in worker
promisequeue make queue length public
update help message for public url option
add default port for server
fix ignored babel files closes
fix chokidar ignored regex
add fallback port to environment variable port
fix hmr hostname in browser
fix hmr bug
isassetinancestorbundles consider bundlegroup nodes when tracing ancestries
add support for amp img src path to img jpg
update pugasset js
update loadparcelconfig js
remove includeall from traversebundle
fix error when scss file is empty
fix circular dep and skip htm test for now
only watch directories on macos
fully fixes make h same as help command
fix sourcemaps failing on refresh
fix detailed bundle report
allows for dots in lazyloaded file names
fix typo in uglify js
fix test option
sort child bundles by asset names to avoid race condition in tests
fix error when env is a folder
support resolving all extensions that are registered
using the bundler via api doesn t terminate all workers
don t extract pipelines from entry specifiers
parses process env port to a number
json always has no dependencies no need to parse it
exit on build error when running parcel build
use the specified host
remove unnecessary fs import
merge pull request from micopc fix publicurl issue
bump test timeout for slow test
fix asset size always be zero when building vue project
display absolute path on failed dependency resolve
skip loading dependencies for assets that don t have import require
pug config debug typo
handle empty config files
make asset flow strict
add support for scoped plugin packages
support vk image
mark ast dirty
use var over let in bundle loader built in
improve check for graph validity
merge pull request from parcel bundler wbinnssmith remove add subrequest
add cache dir option to cli
fix memory leak
make cssnano configurable
resolve all entries and rootdir
allow for null sourcepath in dependencies
always process stylus
refactor new bundle creation
fix serving files with query params
improve devserver warnings
use esnext target with typescript and scope hoist
fix hmr runtime error
update postcss js
remove duplicate graph traversal
add support for chrome extension protocol to bundle url js
respect specified host host headers not just host
merge pull request from nebtex master
merge pull request from kenperkins responseemitchange
emit a proper error
merge pull request from andrewschaaf respect host header
doc remove use of gendered pronouns
accept options in arguments
do not pass timeout to http request
fix bug test didn t catch this though
moving response emit above setheaders on destination streams
merge pull request from fredkschott linting fix empty if statement
do not create buffer from object in setcontentlength
fix the set cookie header
merge pull request from isaacs strictssl
fix just another global leak of i
simplified check for whether response headers location starts with http or https
fix oauth tests add missing key due to bug
don t forward authorization header across requests to different hosts
merge pull request from marsup master
add ciphers and secureprotocol to options in getagent
set href as request js uses it
fix bug in pipe handling thanks tanepiper
merge pull request from sethbridges master
disable tests test timeout js on travis
properly flatten chunked responses
changing so if accept header is explicitly set sending json does not overwrite
typo causing typeerror cannot read property length of undefined
merge pull request from redbadger master
if query params are empty then request path shouldn t end with a
add support for manual querystring in form option
when add realm to the front of the params array do not replace anything in the array
add test for missing request body when using json true
expose tough cookie s getcookiessync
test case for
add test for random cnonce
merge pull request from zertosh simplify defermethod
it appears that secureoptions is an undocumented feature to fix issues with broken server see joynet node
add backward compatibility
checking for ssl fault on connection
let s see how we do with only the main guard
merge pull request from cauldrath redirectevent
add a space after if
merge pull request from isaacs proxy ssl hostname
fix idiotic bug i introduced
ignore null authvalues digest
merge pull request from seanstrom bug callback called twice
setting proper statuscode
skip agentoptions tests when running under istanbul
merge pull request from mridgway gzipfix
add some additional hacks to work in the browser
pause and resume should act on response instead of request
merge pull request from jgautier master
do not try to remove listener from an undefined connection
debug when json parse on a response body fails
merge pull request from nrn master
add responsestarttime timing
add body property to resp when we have it as a shorthand fixes
add del alias thanks tanepiper
merge pull request from nylen tests disable timeout on travis
add some missing mime types
add back remove debug metadata
fix lint error undeclared var i
hang copy of defaultproxyheaderwhitelist on the request class
global leakage in request defaults
remove unused variable for eslint
new http client doesn t require such explicit error listener management
ensure only the properties we expect are there
avoid lots of lookups to process env
allow options url people do it all the time might as well just support it
writing requestbodystream into request
exposing request class
rewrite tests test tunnel js to use tape
lost a forever fix in the previous merge fix
set content type on input
ensure that uuid is treated as a property name not an index
chore ensure piped body is exactly the same
merge pull request from mafintosh master
add tests around using tests for objects in a query string
remove request body when following non get redirects
merge pull request from pastaclub master
remove guard there are some cases where this is valid
merge pull request from jeffmarshall master
reintroducing the wtf escape encoding also fix a typo
fix typo with remaining tests
merge pull request from jaipandya master
fix test tunnel js
make password optional to support the format http username hostname
add drain event and returning the boolean from write to proper handle back pressure when piping
merge pull request from yaxia master
allow empty string
use rejectunauthorized false in tests
merge pull request from scarletmeow master
merge pull request from fredkschott linting fix missing new
typo in request defaults
check for content length header before setting it in nexttick
response abort should be response destroy
merge pull request from scotttrinh null form data
also set content length when options body is the empty string
tests fix test timeout js events unit test
update request js
fix reference error
merge pull request from ifit master
destroy the response if present when destroying the request
merge pull request from finnp headers ref
merge pull request from janjongboom master
remove console log of multipart
test for unescape
error on maxredirects exceeded
merge pull request from isaacs master
titlecase authorization for oauth
copy headers so that they survive mutation
fix headers piping on redirects
fix case where no body exists
merge pull request from medovob secureoptionspooling
add dynamic boundary for multipart requests
don t end response early in tests test pipes
fix the code style
fix remote memory disclosure in multipart attachments
fix multipart content type detection
fix lint error
remove check for connection header as keep alive is now default
add back some missing code remove duplicate lines
fix encoding of characters like
add support for specifying an encoding for the response body
fixes outdated check on options json from before we had boolean support
merge pull request from simov fix default options
add ssl support
fix typeerror when calling request cookie
merge pull request from nanodocumet master
merge pull request from timshadel json
remove redundant call to stream constructor
test getcookies method
add check to see if the global pool is being used before using the global agent
bugfix allow add cookie to wrapped request defaults
add url to redirect error for better debugging
remove tiny white space
improve error reporting for invalid protocols
referer header should point to the host name of the original resource when redirecting
prevent setting headers after they are sent
pass through basic authorization option for https tunneling
default value only if undefined in request defaults misplaced return statement
add pass message when tests finish
merge pull request from kapetan master
fix typo in noproxyhost definition
pass servername to tunneling secure socket creation
remove two more spaces
merge pull request from tobowers master
revert workaround for github issue
comment to explain init and start
fix cookie jar headers cookie collision closes
merge pull request from jplock master
check error type instead of string
protect against double callback
merge pull request from eiriksm browserify changes
fix basic auth for passwords that contain colons
merge pull request from drudge master
fix new params style on master for head request
add src stream removal on redirect
preserve the original full path
handle auth password and auth username
fix defaults when url is passed instead of uri
failing test for setting head with a body
merge pull request from nw master
add unit test for uri specified by hostname and port
lint fix pr from pre standard was merged with passing tests
merge pull request from duereg master
silence eventemitter memory leak warning
fixes missing host header on retried request when using forever agent
propagate datastream errors
copy options before add callback
explicitly comment empty block as empty
fix typeof check in test pool js
use request abort to abort the request when the request has timed out
some explaining comments
forever inherits bugfix
merge pull request from nylen fix localaddress osx
json test should check for equality
remove dangling commas
fix eslint curly error
relaxing comparison to cover differences in error messages
linting fix remove unused variables
add better debug message on failures for get tests
don t require ssl for non ssl requests
merge pull request from bigeasy ignoring
add redirect event
revise setuptunnel a little more
support json apis that don t set the write content type
fix localaddress test on os x
allow header white exclusive lists in any case
linting fix restructure bad empty if statement
introduce multipart chunked flag
merge pull request from apeace master
merge pull request from tonistiigi fix basic auth pass with colon
fix bug where callback fired for every redirect
remove content length and transfer encoding headers from defaultproxyheaderwhitelist
merge pull request from calibr iojs error fix
throw error if method is head and sending a body
add reference to request instance on response to make it easier on inline callbacks fixes
merge pull request from sreuter master
test request content type for multipart form data
fix piping from a file test
made json and multipart options coexist
throw on error
merge pull request from fredkschott linting fix func call in conditional
update the internal path variable when querystring is changed
add newline to end of test files makes for cleaner diffs in the future
fix bytelength string lenght issues
merge pull request from drudge master
fix end bug i introduced being stupid
proper checking and handling of json bodies
if the error is handled and not throw we would still process redirects fixes
change on to once remove possible memory leaks
verify that body is not empty
add additional header test
avoid duplicate emit when using a timeout
add test for request pipefilter
add allowredirect function should return true if redirect is allowed or false otherwise
fix log error
allow parser errors to bubble up to request
don t delete headers when retrying a request with proper authentication
default to globalpool
support key and cert options
silently ignore errors when add cookie to jar
merge pull request from nylen setcookie regression
merge pull request from romainlk master
move the tunnel override flag initialization to the ctor
double quotes are optional and the space after the could be required
improvements to json handling and defaults
redirect test also assert that the request cookie doesn t get doubled in the request for the landing page
run stringify once
escape special characters
merge pull request from mac master
emit error not cookieerror
disable https strict test when running under istanbul
fix setheader with undefined value
cookies shouldn t be discarded when followredirect true
merge pull request from wpreul master
workaround for github issue
merge pull request from simov fix referer
fix linting error
put a test around array serialization
add support to client certificate with proxy use case
remove another superfluous test
merge pull request from cartodb removelistener
modifying json request test
failing test for pipe dst opts
merge pull request from abacuslabs master
remove unused require assert
add more explicit error when undefined is passed as uri or options
make return statements consistent
use once when listening for parser error
try normal stringify first then fall back to safe stringify
damn travis can be incredibly slow
merge pull request from oztu master
change secureoptions to secureprotocol for https request
merge pull request from incredible labs master
set transfer encoding for multipart related to chunked by default
emit an event after we create the request object so that people can manipulate it before nexttick
merge pull request from proksoup master
headers add to http connect requests
fix respect the strictssl flag
fix double var declaration
correction for windows os identification
semi colon fixes
remove debugging statement
merge pull request from joaojeronimo master
rewrite tests test optional js to use tape
merge pull request from nylen fix missing var i
fix spelling of ignoring
linting fix remove function call from if else conditional statement
don t overwrite process stderr write without restoring it later
merge pull request from phillipj debug invalid json
merge remote tracking branch upstream master
reverted null body behavior
merge pull request from jhurliman master
prefer late return statement
clear stderr array before each test in node debug
process exit is fine in tests
export default server ports
merge pull request from isaacs master
add some code comments about why we do the extra checks
jshint creating global for variable should be for var
remove irrelevant comments
fix proxy no proxy false positive
add accept header to json support
request will keep it s own agent pool so that it can expose a maxsockets setting for easy pool sizing
append secureoptions to poolkey
fix travis test
copy the input headers object
fix a typo
fix a couple of typos
merge pull request from nylen fix typos
cleans up test localaddress
merge pull request from spollack master
add multipart test for transfer encoding and content length
change isurl regex to accept mixed case
perform stricter timeout value validation
moving to standard stream api
merge pull request from juul master
merge pull request from jeffmarshall master
improving test localaddress js
optimize do unnecessary processing when changing options to null
multipart requests should respect content type if set issue
do not overwrite established content type headers for read stream deliveries
missing new operator when creating and throwing a new error
fix options not being initialized in defaults method
fix for failing pipe opts test
merge pull request from fredkschott linting fix unused var
add braces to my conditionals
check the value of content length before setting the header when posting a form data
implement parsing json response when json is truthy
emit complete event when there is no callback
use strict mode in request
fix fallback for browserify for optional modules
increasing timeout because travis is kinda slow
fix test emitter matches on req url so it needs the full url
merge pull request from richarddong master
merge pull request from camilleanne master
changes tests to run assertions on tests
emit formdata error events as request error events
improve baseurl error test names
optimize environment lookup to happen once only
resolve image path in har test
rearrange requires to prepare for fix debug code
multipart test check for default content type header
add test for setcookie regression
force all cookie keys into lower case as suggested by linusu
merge pull request from eiriksm browserify compability again
gfm tables do not break paragraphs see
do not tokenize emails as html fixes again
test the minified version with node test minified
make emend const
fix octal literal
update the list of markdown escapes per commonmark
breaking change space is no longer allowed between link text and link label in reference links
fix lint per usual
add parser parseinline
test return proper exit code on tests passing
fix gfm paragraph rule to account for tables
remove test index array code
a more accurate del rule
do not escape style elements
html tests werent passing better solution here
fix unit tests for slugger
perf get match indent to code
escape lang class
move tables to gfm
update lexer spec js
fix regression caused by merge conflict
ignore angle brackets in ref links fixes
throw an exception when parsing with no links object
merge pull request from tomtheisen gfm strike
re fix parsing of empty code blocks after the previous fix
perf throw error if raw undefined
clean up code
fix table parsing closes
merge pull request from guybedford master
update src rules js
allow version to be a github commit
as usual fix my lint errors
build skip ci
minor async refactor
fix where i accidentaly overwrote a change upstream
fix browser test
fix small problem in def rule see
fix incorrect compare rule
fix blockcode render more test cases passed
fix angle links
allow colons in tag names and attributes decrease group depth in regex
fix variable name
handle mailto differently
call paragraph renderer for stray text tokens
add hard line break when backslash at eol
update lexer spec js
style after lint
fix line break
add collapsed reference links links links
fix hr rule
save current link parsing in pedantic mode
allow naked sqaure brackets inside link text closes
match original markdown behavior for headings
treat sanitized block elements as paragraphs fixes
merge pull request from insanehong master
revert number of capturing parens in
fix hr in list rule rename rule
style after lint
remove around pedantic title
remove old sanitize check
merge remote tracking branch forbeslindesay execption when no links
pass error to callback on highlight error fixes
ignore links without urls closes
add preliminary smartypants code see
do not allow control characters in link destinations per commonmark
load default options in demo
merge pull request from mkozhukh master
code blocks don t ignore trailing newlines at the end
merge pull request from jgermade master
merge pull request from davisjam sonatypereport
we are talking to the developer that this type of token can not be found
merge pull request from lwansbrough master
check for href property on links
allow lazy blockquotes
add checkbox to next token in loose task list
fix when text is blank
improve tag to ensure it excludes autolinks always
avoid backreference in link regex
move def lower in paragraph rule
fix marked can t run without options parameter
merge pull request from uzitech show failing by time
minor remove pointless assignment
encode urls with xx entities
add html to textrenderer
save renderer to inlinelexer options
simplify tag regex
update tokenizer js
em bug fix
strikethrough support for gfm
better loose regex
declare undeclared variables
update lexer spec js
list regex change
allow escaped or balanced square brackets in shortcut reflinks
more conformance for blockquote
email address domain must contain at least a dot and must not end with dash or underscore
paragraph rule link defs cannot interrupt a paragraph
send inline html to renderer
stop using my own personal version of showdown for benchmarks
perf regexp for em close
modify lheading converting for support standard markdown syntax
perf to save the contract
add inline html test
update copyright year
update src rules js
fix italics modifier close issue
lint make the outer function a recognizable iife
merge pull request from evilpacket no vbscript
revert allow dashes in language names fixes
fix marked version
don t allow foo bar to be treated as link
allow parens around link reference definitions closes
add deprecation warning for sanitize option
make strend const
fix issue when using custom requirejs loader
remove console log
move table rules to correct positions
fix options not being passed
refactor html block rule
test re activate passing tests
fix inline code regex
export textrenderer for convenience
test only remove one space
merge pull request from mithgol exit code
add newline to space test
limit ordered list marker
allow frontmatter in original tests
merge pull request from uzitech demo master version
add ability to escape
return null on non string input
link def labels cannot contain balanced brackets
use default options in renderer by default
update lexer spec js
clean up code
merge pull request from mccraveiro blockquote
fix test names
small regex changes
fix inline em
readd slice to test index js
re enable passing test
escape html entities in link image href
switched out tabs for spaces
allow redos spec only
make escapable closes
update lib marked js
merge pull request from mccraveiro ordered list
fix blockquote paragraph continuation
md files can have skip and only to help debug
remove unnecessary if check
add marked origdefaults
add back q mark
allow spaces in urls
update percent complete
fix test failure
remove trailing space
add amd support when define and define amd detected
more conformant loose list check
add do not edit comment in compiled files
fix backtracking vulnerability in block pedantic html as in
convenience parse api
string prototype type text scares me
fix whitespace after tag
allow dashes in language names fixes
remove focused tests
deleted an extra line while remove comments
fix html comments again fixes and
move vertical bar escape to gfm only
updated inline grammer regexes for strong and em
remove tabs in table output
allow a regex source string as a parameter for edit
don t allow spaces in link uris
closer to the original
fix strong overlapskip
clean up spacing for rules
fix changing global defaults
angle bracket links
add assertion to existing test
fix code blocks
makes link text show up even if link is dangerous
skip tests that are known to fail as enumerated in chjj marked
avoid code duplication with regexes
fix hrs in list items code fixes
style after lint
more compliant blockquote rule
revert implement smartypants
allowing dot before extension
update src tokenizer js
fix paragraph rule
allow html without n after
allow test flags to be more than true or false
security fix unsafe heading regex
remove empty space from lines
always sanitize html blocks even if pedantic benchmarks
lint marked js
support older regex engines
allow headings with no trailing newlines closes
space after bullet
use regular quotes
fix bug cannot break line well
security fix possible redos vulnerable regex rule by refactoring doesn t change the alphabet
remove when is merged
treat escape token same way we treat plain text tokens to allow their transformation in renderer
update lexer spec js
print more descriptive error messages when source is undefined as in
fix closing delimited close issue
show items in lexer data
node test should have non zero exit code if some tests failed
test perf torenderexact method
add comments for clarity
link labels must have at least one non whitespace character
require marked relatively for tests
remove g flag
only match http https in gfm autolinks
test marked parseinline
allow nested code closes
update fences after tables regex
simplify table regexes possibly use backreferences for leading whitespace
single tag blocks
gfm supports arbitrary characters in fenced code block language identifier
runtests should return true if tests passed false if some failed
allow differing bullets in lists closes
disable gfm for old markdown pl test case
update src rules js
default top to true
fix nested links img
prevent catastrophic backtracking on em rule fixes
do not process inline markdown in pre elements
allow matched double quotes in link definition title markdown pl behavior
output newlines closes
add optional semicolon in html entities regex
blacklist vbscript fixes
refactor item bullet list regex
more compliant lazy blockquotes closes
completely escape fenced code
remove on heading regex
add lexer lexinline
different link error handling
add potential fix
fix small typo occur r ed
merge pull request from zaggino master
remove unnecessary trim
fix new line after table
use faster text rule
ignore inline elements for self closing block level html tags
never allow hr in list items as per markdown pl
handle explicit mailto in autolinks
drop other em strong behaviors
remove passing tests
filter complex objects out of json defaults
returning marked in setoptions to allow for fluent interface
merge pull request from eirenliel master
option pedantic overrides gfm and turns off gfm tables and breaks
allow matched double quotes in link definition title only in pedantic mode
remove bench debugging
clean up new module check
make sure block isnt too greedy
merge pull request from kostyatretyak patch renderer
make list item regex consistent whitespace
add support for manual text node rendering
fix html comments fixes and
remove invalid failing test check
explicitly send a false top state
merge pull request from niggler master
fix inline token error
fix nested em strong
fix backticks in inline code fixes
remove commented line
update marked js
add gfm code blocks
trim inline code fixes
quotes in link defs
add highlighter as an option
add marked use
remove leftover debuggin statements
remove underscore require from app js no longer used
add new tumblr token field
changed mongostore db property to url
add linkedin auth and auth callback routes
add env var for linkedin callback url
remove unnecessary route callbacks from facebook and google auth
add a failing test for travis ci
fix bug in remove providers
remove console log in user model
remove unused paypal logo from img folder
updated message if user with that email already exists
fix twitter auth finduserorcreate
github authentication routes and configuration
fix a typo
use app get env
minor improvements to generator
enable express compress middleware for gzipping content
hack to exclude internal links from the list
add facebook email permission to route auth facebook
use https profile image url in twitter auth
override mongoose promise to global promise
updated flash message in google strategy
remove red method from console log message
changed to req originalurl as req url is not a native express property
merge pull request from nacimgoura github
remove unnecessary default empty string from profile fields
api this fix
replace broken show function in github api example
remove express error handler development only
remove dynamichelpers comment
reverting for if else
paypal urls must be taken from env
add comment blocks for twilio get and post controllers
fix returnto with queryparams
renamed template account to profile
updated the message on remove authentication generator menu
fix login partial issue
remove windows if condition
mapped sengrid error message to msg property so it matches express validator
add a flash message your account has been deleted
update nodemailer api usage in contact controller
created username availability controller
placed app express right after module dependencies
send user instead of req user
add comment for post api twitter controller
add get forgot test
fix foursquare secret token
stripe post to api stripe uses source not card in the stripe charges create method
prevent remove email during update profile
use instead of undefined
merge remote tracking branch origin master
add ordr in keys
renamed referer to referrer
updated linkedin auth routes to include mandatory state object
updated schema s default values for password token and expires fields
update model js for syntax error
fix flash message in case of a rare database issue
pass user object to aviary last fm and nyt api pages
stores lowercase email for gravatar
merge pull request from starcharles master
updated error messages
top level index layout renamed to home
oauth tokens are now explicitly defined in the model schema
remove helpercontext property from connect assets
replace fat arrow with function keyword in user model
update secrets js
add csrf security measures on login forms
make flash messages available to every view
add isauthenticated and isauthorized on linkedin
merge pull request from reesretuta master
fix yahoo weather api example
add callback to async waterfall for error handling via express middleware
add a brief comment explanation of how passport strategy works
update bcrypt rounds
updated linkedin secrets js to have scope clientid and clientsecret
merge pull request from deiucanta paypal fix
refactored gravatar mongoose method
success notification and redirect back to twilio api page on successful text send
add email validator
handle account verification mismatch
convert email to lowercase in localstrategy
trivial doc comment format fix for twitter auth
remove redundant req assert for empty email
cleaned up middlewares
remove res locals cachebuster cache busting is done by connect assets
merge pull request from nacimgoura uniform
update returnto middleware
fix github deprecation warning
change test to passing
user errorhandler in development only
merge pull request from robts master
updated linkedin error flash message
update flash message on successful forgot password request
add support for assets concatenation and minification
fixes can t set headers after they are sent
moved last fm key secret into config js
profile data defaults to empty string to prevent value of undefined on account page
fix redirect issue in the account page
merge pull request from raidok master
add passreqtocallback to tumblr oauth strategy to get currently logged in user
remove extra app use express csrf
disabled hackathon starter generator on windows platform until further notice fixes
only two stripe routes get and post api stripe
add gender field to facebook strategy
fix bug in password reset
add authorized middleware to github route
merge pull request from dsavenko master
add missing tokensecret when new twitter account is created
updating twilio api
updated mandrill demo login and keys
show the user the error message instead of discarding it
fix a bug with connect assets where assets were served twice
updated token unlinking
merge pull request from yasharf master
instagram auth generator checked by default
merge remote tracking branch origin master
fix issue in the web scraping example
update post forgot description
session management improvements
fix a bug that didn t remove instagram auth properly
remove favicon middleware
fix spelling mistake
fix owner is undefined
fix returnto redirect logic
update github api example
remove mongoose reference
merge pull request from notjackson remove typo
refactor gravatar function
changed updatepassword controller validation to use express validator instead of custom if statements
remove db validations
merge pull request from empurium master
add clockwork demo api key
add success flash notification on successful password reset
adds paypal credentials for api configuration
updating email on profile page will correctly update it in the databse fixes issue
merge pull request from rockabe gravatar
user model doesn t seem to be used in this file
updated tumblr blog url
handle email password login attempts for accounts with no password
sparse indexing on username field to prevent duplicate null username error
add user password test
merge pull request from xasos master
remove flash messages view helper
add successredirect on google auth
fix eslint issue
updated to nodemailer field in reset controller
post signup controller updated username to email field during new user creation
merge pull request from garretthogan twilio api update
fix hacker news api web scraping example
remove unique and sparse properties for oauth providers since id is already unique fixes
improved mongodb connection error message in terminal
update secrets js
missing public profile and email after switching to another facebook app id and secret
minor improvement in contact controller
remove yahoo api route
connect mongo will now use existing mongoose connection
fix typo in message
adds paypal routes for the sdk
generator expects all mail service provider s configurations to have the key user
flash error when updating email to email that is already taken
send validation error to client if any of the required fields are missing
redirect to original url when users sign in via local strategy
remove partials route old
add linkedin id field to user model
fix for mongoose url string parser deprecation warning
improve security by remove x powered by header
user currently logged in user s facebook id instead of hard coding
pass session object directly into connect mongo
update sendgrid email address
updated test for creating a duplicate user
code style formatting
updated user schema provider field was not used
merge pull request from generalzero master
merge pull request from nacimgoura master
updated twilio credentials
updated instagram generator
fix express session deprecation warning fixes
update mongodb connection error message
updated contact form mail options
updated to and subject nodemailer fields on forgot password controller
add title forgot password to get forgot template
move session secret in to config secret file
flash notification on oauth provider unlink
remove mongoose global promise
fix instagram api token issue
remove todo comments
nonexisting routes will simply return an error page instead of redirecting to home page
adds link when developing locally on startup
fix twitter generator strings was google instead of twitter
sendgrid now uses a more sophisticated email object rather than plain object more flexible for sending complex email message
add mode type to console log when you start the server
add sass sourcemaps
fix object object displayed on profile update success instead of the actual message text
merge remote tracking branch origin master
config mongoose to use the new server discovery and monitoring
add forgot your password routes
routes for the rest of apis
setting foursquare api version and mode
add note about firefox broken sandbox support fixes
finishing my sentence
fix link to article
merge pull request from patrickkettner vml notes
made minor changes based on tips from stucox
add check to verify that enableclasses has been initialized
remove unnecessary metadata
fixes edge regex in uaparser
update feature detects gamepad js
fix metadata path on windows
better webgl extensions output
fallback to checking drag and drop events on element
add test for required attribute
add test for autocomplete attribute
merge pull request from patrickkettner flash fix
correct reference to testprop
add doc block with description of how the test works
merge pull request from whawker master
userdata test more metadata
add a few more globals to the jshint lint options
opendatabase safety net for incognito safety net
remove random f character sorry about that
drop the parens
offline to applicationcache
fix tabs spaces in typedarray feature detection to pass linter
add escape hatch for browsers that don t support autoplay at all
return test metadata
paul you should always lint before you commit
separating var declarations
fix localizednumber test overwrite the existing value rather than appending
update oninput to use hasevent
go back to using bind since it s used other places anyways
prevent warnings in ie
another technique for flashblock detection
update supports check to look for the cssrule rather than the sub property of csssupportsrule
docs on exif orientation ref
move jshint out of the build task to improve build speed
webgl test add now that spec and implementations stabalized fixes
exagerated font size
convert webp lossless test to a non alpha image
code style cleanup use single quotes
catch exception when running in headless chrome
fix typo in documentation and capitalized ie
remove font face tests from test suite as it is no longer async and has reducd api footprint
remove dataset caniuse comparison at least until fyrd caniuse
was returning undefined in non supporting browsers change to return actual boolean value
tweak to boxsizing test needs to test camel case thanks snobojohan
fix for safari s false positive on datalist
update hairline js
fix a scoping issue with multiple modernizr on callbacks for a single test
extra note about msprefix not being msprefix thx erik
localizednumber exit early if other tests fail
add test for maxlength attribute
add test for pattern attribute
updated background clip text test to use testallprops
merge pull request from patrickkettner displaytable typo
merge pull request from patrickkettner noalpha lossy webp
remove camelcase for new properties get low get low get low get low
corrected name for touch to touchevents in modernizr caniuse mapping
we re aware that we re creating functions in a loop
updating to reference mathjax contribs prior discussion and myself as requested by paul
merge pull request from tnajdek master
output elem test rename to avoid class conflicts incl our test suite
dry up some vendor prefix action
fix masks test so that it doesnt count firefox s svg mask result
property is displaytable not display table
merge pull request from philippsimon increase mochatest timeout
update feature detects exif orientation js
less string concatenation
merge pull request from modernizr add info css backdrop
add test for list attribute
straight boolean return value on svg test
fix link to kangax s iseventsupported
make phantomjs exit out of caniuse data matching early as it s unecessary
remove extra style element from head fixes
browser support deets on css mask
updating to use modernizr prefixed foo bar syntax
make sure test flash embed is remove when its been wrapped
webdatabase to websqldatabase references
small fix for svg filter test
test for draggable attribute
bugfix prevent viewport jumping when using css hyphens
clean up async metadata
test suite minor webgl may not be a pure boolean thx to recent plugin addition
ignore bad caniuse data for firefox fixes
merge pull request from deschtex master
sleeker indexof evaluation minor
merge pull request from zhorvath master
rename test to inlinesvg
merge pull request from josh draganddrop
expose some more apis modernizr event testprop testallprops addelem
merge pull request from dbow cookies
catch exception when firefox has dom storage enabled false
no js to js revised smaller
improve custom protocol handler js
merge pull request from philippsimon websockets fix
merge pull request from smyru master
suppose this isn t needed anymore
fix issue css scrollbars false positive in ff
touchevents use mq in touch event detection
use just false instead of function return false if the test already failed
add metadata for lastchild selector test ref
add canvas dependency to apng test
moved modernizr pointerevents modernizr csspointer events api break
merge pull request from bastimeyer master
modify css exclusions to css shapes since we split up the specification update references
xdomainrequest simplify detect per comments in
correcting typo in intl detect property name in metadata
remove memory leak precaution based on
make form validation output valid code
no console log in addtest
merge pull request from stucox master
fix a bug where the generated flow name was invalid
box shadow test
rename modernizr file to modernizr filereader
check if values are actually false bool objects before setting css class
remove offline detection via online
improved behavior based detection
fixup test suite for better ie reporting
minor hasevent test suite tweaks
fix websockets test on some panasonic smarttv browsers
catch window find exception in ff for android fixes
merge pull request from modernizr stricer inference
making sure the constructor is nice gzip does the rest
merge pull request from stucox master
khtml prefix tested
save the whales and or the output of the build
mathml fix missing amd dependency on teststyles
additional webview check add to fileinput test
fix test for checked pseudo selector
change abspath to path resolve to fix windows node issue
updated notifications with modernizr prefixed nofitications window
merge pull request from ryanhughes firefox css hyphens
syntax leftovers spoiling the fun
add test for multiple attribute
grunt banner version is out of sync fix
updated comment describing interface to setclasses
add sticky positioning knownbug fixes
fix style issue
merge pull request from modernizr note blackberry issue
media type support for audio video rejigger for opera compat fixes
test suite has some laggy tests yeahhh
merge pull request from jnordberg master
supersecret microsoft requested sensorship
merge pull request from patrickkettner border image doc
changed an existance check to an in to avoid exceptions being thrown for disabled features in ff fixes
merge pull request from patrickkettner supportoldsupports
add metadata to modernizr framed
remove outdated firefox bug from gamepad docs
add qunit to default grunt command
customizr is currently failing testing due to this property
detect input autofocus
note add about androids weak keyframe animation impl cc daneden
exit with non zero status if there were failures
fix issue and issue
webkit likes textarea value like innerhtml but only after you force a reflow this is crazy
fix for typo in indexeddbblob test preventing it from running
bumping version to pre
extra check on localstorage should address
whitespace crlf s
fix code style
add note to cssgradients test re false positive in webos fixes
merge pull request from patrickkettner updated svg filter
add test task to alias qunit and jshint
update websockets detection
remove saucelabs testing form travis until it passes properly
remove some no op code in webgl js
clippaths test is back
merge pull request from patrickkettner dragndrop docs
correctly define prefixes fixes
add metadata to cli
merge pull request from stucox master
remove unused module
history apis should be false on file uris
merge pull request from josh geolocation disables webkit page cache
add an alternative check to detect opus codec support
update detect for fullscreen api fixes
update meta data of siblinggeneral
add semicolons to avoid generating typos
merge pull request from modernizr autoplay fix
merge pull request from patrickkettner opus
trailing comma like a chump
update inputsearchevent js
finalizing touch detection
new flexbox test got shuffled because the property was renamed fixes
towards passing tests taking into account more tests that follow rules a little less closely than core tests
fix webanimations property
fix vh unit issue on ios safari android chrome
merge pull request from patrickkettner blacklist android transforms
more clarity around web sql database s need to create a dummy db
fourvalues shorthand bgpos detect
update supports js
fix linting err
fixes websockets are now unprefixed
fix double doc entry
add test for step attribute
use default styletest for transition
correct note from within serviceworker js
fix useprefixes for some detects
update webgl test to include supportscontext
reduce the iseventsupported calls for dnd test fixes
merge pull request from carylandholt master
forego object check
add opus detection fixes
shrink webp detect by a few bytes
merge pull request from patrickkettner autoplay note
emoji devicepixelratio vs backingstorepixelratio fixes
merge pull request from dannyjoris backgroundblendmode
merge pull request from obeattie master
remove asterisks from warning text in touchevents metadata because it was messing up parsing
protect against console less browsers in test suite
fix leaky f variable thx karl swedberg
merge pull request from joshuawise master
extend jquery support fixes
fix typo flexbox flexwrap
notes around try catch on websqldatabase check
performance improvement detect api implementation first extreme important for old ies
update invalid datachannel detect
only set the classname once minor minor perf advantage since this script executes in the head
merge pull request from patrickkettner twonottwenty
fix the webgl detection for browsers that support it but computers that don t have supporting graphics cards
updated ogg mimetype to correct value
update autoplay js
actually detect the proper web animations api
check for property and value support for css custom properites
merge pull request from patrickkettner noalpha lossless webp
slice is undefined in bind implementation
remove unused variables
merge pull request from patrickkettner update async metadata
optimize test arrangement fix typo
comment wording tweak
set ascii only option in uglify fixes
add webm codec detection thx mpilgrim
merge pull request from rowanthorpe fix missing semicolons
indexeddb fix invalidstateerror and unknownerror in firefox
merge pull request from ausi csspointerevents improvement
making sure contextmenu test can be run in the head
fix failing test and linting issues
update modernizer flexbox to support both wd and lc implementations fixes
merge pull request from jokeyrhyme mathml amd
fix another bool comparison
update database check for incognito windows
add file comments how it works and why would you want to test that
make note of font face false positive in camino webos fixes
replace text justification text justify
merge pull request from stucox master
fix bad metadata structure
merge pull request from zhorvath updaterefs
merge pull request from patrickkettner move jshint
remove exit it can cause upstream processes to bail too soon
merge pull request from patrickkettner stickynotes
swap testallprops api lines typo whoops fixes thx adambrunner
ensure elem parentnode exists before remove its child
updated spec link fixes
support mozilla css gradients proposal fixes
fix a missing var which was causing linting to fail
remove extra callback param to the testprops methods unncessary it turns out
improved variable assignment in details feature test
remove pre from shiv version
merge pull request from patrickkettner csp update
fix a missing dependency
merge pull request from stucox touch alias
merge pull request from patrickkettner registerprotocolhandler
typo body to html in the inline documentation fixes
merge pull request from hay master
websocket to websockets
bail out of test early if autoplay isn t detected on the video element
alias svgasimg to svgincss
if the test add via addtest is not a function or boolean modernizr dies
detect input placeholders
fixup lastchild syntax fixes
fix this refers to the global object
strict mode caniuse map rename
smaller footprint webgl feature test recomended by chrome webgl team
merge pull request from grayghostvisuals autoplay pulsation
docs on hasevent incorrect results fixes
drop the console call
positioning modernizr absolutely to enable more precise offsetleft check
opacity check fix for german polish chrome floating point bug fixes
kill that nasty jquery global that gets created
only test webkit for their legacy gradients syntax
when binding a function to a context check that it actually has a bind method first
merge pull request from rxaviers minor
sharedworkers got messed up somewhere
update video js
avoid activating geolocation in webkit
fix invalid test background size cover
changing test names to lowercase minor update to structure of apng test case change for getusermedia
merge pull request from stucox sc dev
cleanup so jslint doesn t bitch so much
modernizr touch add
merge pull request from paulirish overlayscroll
add bug notes about dragndrop on mobile
fix docs typo
add tests for max and min attributes
remove blacklisting of old android from fontface js closes
amendments mentioned in
revert new flexbox test got shuffled because the property was renamed fixes
merge pull request from stucox master
phantomjs tostring s a window different than chrome does
add test for file api
use cached canvas test to avoid running it twice thx phiggins slexaxton
remove cast from promise test
make sure activexobject exists before accessing it fixes
add a bug note to localizednumber re dependence on localisation settings see
shorter webp data url thanks to derschepp
add textshadow to core fixes
tighten up emoji s detect for canvas for edge case false positives
add notes to the vml detect
update incorrect metadata
fix exception in indexeddb check on safari
merge pull request from alrra line ending
improve contenteditable test
add notes to smil detect metadata
fix a typo in testmediaquery
fix testprops so it works on strict xhtml browsers
made check a little clearer remove a nested if statement
update stylable scrollbar detection using teststyles
add in dom test metadata
contains includes string prototype
onhashchange checks for compat mode
update the stripdefine to the new r js output
fix metadata path
minor docs add discussing textarea placeholder pointing to the feature test fixes
remove trailing comma
fix linter err
generate meta js now writes the metadata to dist metadata json
merge pull request from philippevk master
reorder video autoplay videos
the addtest function didn t test for the variable enableclasses
fix grunt build
add hyphen less alias for flexbox legacy to allow use of dot notation in javascript code
webos false positives on the boxshadow test fixes
merge pull request from patrickkettner oldffsandbox
use modernizr canvastext instead of reinventing the wheel in the emoji feature detect
extra info around flash detection
fix for check for chrome frame in download test
merge pull request from lennym master
fallback for touch test should fix any chrome false positives if we miss them
metadata doc block add
remove unnecessary var declarations
fix for input formtarget detection t in target must be uppercase like formenctype and formaction already are
fix possible unintended scrolling in oninput test
merge pull request from patrickkettner border image doc
merge pull request from patrickkettner silkissue
tweak testprops js documentation to avoid misreading
fix typo fixes
add knownbug for blackberry and input multiple
omment sentence change
the addtest function didn t test for the variable enableclasses and it should
fix csshypens detection on firefox and safari
tighten comments link to thread
making sure docs are good as is
add metadata to workers detects
merge pull request from byk websocket update
move generateinit deps inside function scope
clarify reasons to ignore navigator cookieenabled when detecting cookies
addt l docs on cc conditional comments and their unreliability
fix invalid property value in the getrandomvalues metadata
add comment about webkit geolocation bug
merge pull request from patrickkettner fix useprefixes
remove touch alias for modernizr touchevents enough is enough
documenting textinput false positive fixes lol fixes
merge pull request from dmolsen fix contextmenu
merge pull request from zazabe audio codec opus
add comment documenting the purpose of the heartz string closes
false positive for csstransforms test
fix camelcase in userselect feature detect
prevent mixed content warning with font face test regression fixes ref
typo body to html in the inline documentation fixes
improved touch detection now it works on firefox for android devices
minor touch string
merge pull request from stucox master
merge pull request from viljamis master
merge pull request from steveyken master
merge pull request from alrra correct notes
clarifying my comment re document cookie restrictions
merge pull request from scottjehl master
add a comment re implementation of synchronous svgasimg detect
add note about the silk browser having a false positive on box shadow
prevent request loop on opera mini servers
fix pointerlock api test
fix minor typos in comment block
add author doc notes to backdrop test
merge pull request from patrickkettner flash fixes
made tests for appcache support more thorough
add fileinput test mobile ua blacklist for issue
add comment to shapes test
js class should be add regardless of no js presence
merge pull request from stucox master
better way of doing the formattribute test
improved csspointerevents test
remove deprecated apis crosswindowmessaging historymanagement
fix some memory leaking
slight tweaks to the contenteditable test
fix metdata urls
fontface test race condition addressed
merge pull request from dbaron master
small cleanup thx webstorm
background repeat test works for opera now but false positive in webkit hilarious
remove old comment
fix dependency name ordering
allow for use strict in requirejs to fix the strictmode test
use matchdep to detect if devdependencies are installed require each if so
merge pull request from bobthecow master
wrapping cssregions detect delta calculation in a parseint fixes
merge pull request from brendankenny master
update script header
table cell test remove unncessary createelement shortcut good call slexaxton
register search key shortcut for help overlay
fix animations intermittently not triggering in firefox
merge pull request from archive master
merge pull request from spy seth dev
server notes plugin no longer syncs overview mode
merge pull request from reyerstudio feature shortcuts dev
sync starts playing new embedded content
update phanomjs pdf printing to work with latest pdf css
update deserialize regex to work with decimals
add ability to prevent swipe for specific elements
allow data background images to load even if there is whitespace at the end
allow to be called directly before or after closes
allow override of html escaping in code blocks
fix slide transitions in ios
add option to gruntfile js to specify server port
allow switching off preview links
don t trim aria status
alias notes alias for md speaker note separator
add autoslidemethod to default config
update reveal js
fixes bad nan applied to style on updateparallax
attribute values should be quoted
cleanup classes when backgrounds are synced
zoom js uses ctrl click for linux alt click for others
fix socket id
for a fragment allow to show a separate note defined in it
correct target for cssmin
remove needless semicolon
add logitech presenter tools black screen button compatibility
merge pull request from lechten registersearchkey
disregard v index when there is no vertical slides backgrounds in getslide getslidebackground
only use zoom to scale presentations up fixes shifts in text layout
only read textcontent for aria callout
remove calls to missing modules from gruntfile
merge pull request from jjallaire ontouchstart nullcheck
null check for notes in markdown parser
relax keyboard blocking condition
bugfix continue autoslide after resume
fix slide notes printing
allow background images without protocols to have query hashes parameters
add the remaining gulp watch
merge pull request from rparree hash problem
config option for disabling the help overlay
fix transitions in firefox
merge pull request from fil master
test for getprogress
updated improvement on spaces and indentation
merge pull request from frockenstein master
fix issue where syncslide would not run play newly add background content
simplify logic for finding slide backgrounds
merge pull request from christian classics ethereal library squarepresentationlinear
open notes in new window instead of printing to console
merge pull request from pierreozoux master
include all themes in grunt task
add version comment
fix function name
fix contenteditable check
avoid stripping out notes when parsing markdown closes
better error reporting for external markdown files
allows to specify port from env on multiplex plugin
fix detection of named links closes
ignore metakey modifier too
add nextfragment and previousfragment to the api
update reveal js
widen watch for html files required for updates to demo html
merge pull request from tristansokol tristans add useavailableport
dont close dev server when tests finish
use current host port
load all images directly when in no transform mode
bubble has light background to reveal container
use internal pointer for current slide
increase test timeout
add mobileviewdistance config key
ontouchstart null check in addeventlisteners
limit scope of mathjax rerender
add custom charset support for external markdown file
revise readurl to handle one based indices
patch to allow multiplexing on url load
changed custom key binding config properties to use keycode instead of code
fix previewlinks target
hide slide number element when disabled
create a single fragment event per fragment index
merge pull request from traff master
fix progress bar clicks in rtl mode
adds support for auto escaping html in code blocks
use home and end key
format note content when creating slide
remove unused argument
control arrows fix visibility bug in linear navigation mode
open slide notes window automatically
remove unused isprintingpdffragments function
merge pull request from jasonkiss feature accessible speaker notes
only announce current fragment to screen readers
docs fix simple typo transiition transition
ensure default can be prevented
reveal container size determines progress bar width
fix notes timer when speaker notes window opened via revealnotes open
don t count slides with class uncounted
add togglehelp function
remove old inaccurate comment
fix notes js to account for upstream updates
merge pull request from jjallaire ontouchstart add nullcheck
better version of data trim on code snippets
remove global flag for replacing whitespaces
allow popup window access to reveal api
lazy loading fallback also considers iframes
merge pull request from rhudea fix fragmenthidden
add pdf ready event dispatched when the presentation is laid out and ready to print
pass through key event when calling keyboardcondition to allow conditional function to filter on key codes
flatten slide number when there are only vertical slides
add missing condition for recalculating scale
fix previous element selection
validate named links according to html id spec
merge pull request from basecode patch slidechange
avoid exception when highlighting empty code blocks
make background videos work on ipad
ensure postmessage data is a string
improvement on spaces and indentation
fix spelling error in print pdf plugin
fix bug where left right arrows appeared when there were no horizontal slides
new syncfragments api method
make sure remotes works without notes plugin being loaded
fix build error
support negative values in query config overrides
fix markdown typo update comment
print port when multiplex server starts
remove extra whitespace
inlined method used only once
fix viewport overflow in ipados safari
broadcast socket id as part of slidedata
adds useavailableport option when serving page with connect
define available values for showslidenumber
self execute data markdown
not got github
improvement of code now none flag is used but the readystate property from video element
syncfragments now returns all affected fragments
merge pull request from tehdmitry dev
fix mixed indentation
kill yank mistake
change transition defaults
merge pull request from opdavies sass extension
merge pull request from suva dev
merge pull request from gothma mathjax md
include layout for notes outside of pdf exports
error check to prevent exception from cipher
make work using replacestate
add base option for grunt serve task
use getelementbyid when looking up linked slides
zoom js works on main wrapper element
linear navigation fix showing arrows when last slide has vertical children
creates a new branch for the feature
background repeat position to parallax background
check window ontouchstart null required for qtwebkit
only use zoom to scale in chrome
autoslide duration with playbackrate
update visibility of slides as part of sync
add isfirstslide and islastslide functions
notes plugin callrevealapi wants apply not call
add more support for external markdown
fix jshint test
merge pull request from commadelimited deactivate nav during pause
default to slide transitions
change event listener syntax
merge pull request from richardwestenra fix missing modules
adjustments to initial node messages
allow tex parsing in code tags
fix markdown js slideattributesseparator selection
set data paused by reveal attribute only for the media elements which are not already in paused state
prevent default on space press
always show media controls on mobile devices
zoom plugin now works for scaled presentations
add override parameter to togglehelp function
alt zoom only works on slide content
use first fragment s data autoslide value when there is no current fragment
more natural zooming on block level elements
serve notes multiple index document as html closes
solves early access error with video element
remove pointless call
revert remove pdf height offset
only call embedded content bits if the slide actually changed
merge pull request from riezebosch pdf print stretch
support sass extension for custom themes
support query parameters within notes plugin
merge pull request from rohithpr notes key binding
fix callback calling when using from a parent directory
property length is not a variable
remove assignment to indexv and indexh to trigger onslidechange at startup and onhashchange
prevent pages from overflowing when printing to pdf
expose navigateprev navigatenext
continue auto sliding through fragments on last slide
fix autoplay of inline videos in ios
get rid of event listener warning when running tests
simplify pdf layout
resolves allow display property to be configured
do not add video backgrounds to speaker notes
refactor code to match new background dom structure
merge pull request from matthewlein master
merge pull request from mbotsch fix readurl
fix current fragment index check when multiple fragments have same index
pdf background size rounding error
add event listener fragmentshown and fragmenthidden
p for previous n for next
update instructions printed on server start
more verbose error slide http necessary
merge pull request from demoneaux background images
made one based indexing optional
don t autoplay background videos in overview mode
fix numeric presence condition
merge pull request from lechten export isspeakernotes
merge pull request from tuurdutoit feature mobile view distance
remove pdf height offset
fix prevents loading remote plugin several times when notes are enabled
fix watching for markdown files
simplify query match
better test for fragment indices
remove unused dom lookup
issue non async scripts were not executed before reveal started
gracefully handle duplicate slide id s
dont show notes container when printing pdf
tweak formatting to match rest of slide
fix search replace error
add public togglepause access
fix issue with background images on first vertical sldie
fix navigateprev past beginning of stack fixes issue
fix issue with links on touch devices tap wasn t working
allowing autoplay in data background iframe for audio and video elements
first implem for add classes to enclosing elts
prevent repeated initialization
uses the right method setattribute
api expose getcomputedslidesize for plugin developers
fix race condition by updating backgrounds after slide visibility
fix iframe unload in firefox
more specific targeting for pdf printing hack
updated md slide separator regex
prevent missing present class when navigating to same slide twice
make speaker notes keyboard accessible
fix an initialization bug when turning off controls and progress in initialization
update plugin markdown markdown js
prevent additional inaccurate showslide calls
correct variable name in inline comment
fix config keyboard now disables s key
more markdown plugin refactoring
add slide width height css variables
refactor fragment pdf exporting to support multiple fragments with same index
add zoomkey option for
fix lazy loading bug related to data background image attribute
bug fix when the first section has a background video the vido do not execute now it s work
adjust check for focused text inputs
move speaker notes into page container when printing pdf
merge with minor tweaks
remove unused dom reference
merge pull request from rollandf master
test same conditions for both getslide and getslidebackground
slightly refactored slidify flow
merge pull request from robertbaron background whitespace uri
remove issue comment
display stretch images in overview mode
fix not bing able to follow same inner links twice with history false
add support for wireless presenter
include search button icon
expose version number through
fix the focus when list is closed on the mouse click event
fix memory leak in attachbody
fix norwegian inputtooshort message
support null undefined false for data to unselect fixes
fix placeholder styling broken in a previous code cleanup closes
fix jshint line is too long
fixes undo changes made to labels during init
merge pull request from brettmas master
fix body element caching for non attached elements
fix triggering of unwated change
take the outerwidth of the element when sizing so we include padding and border fixes
close before destroying fixes
update de js
clear search when val is called closes closes
merge pull request from jdecuyper feature avoid removal of locked options
fix for maximuminputlength
fix string used as regular expression for id attribute
fix ie detection logic
additional tweal to to make formatters skip nulls
fix selectonblur and closeonselect combination
merge pull request from outlivier master
dont bubble up esc keypress fixes
add missing diacritics to diacritics js
break long lines
correct typo in noresults
fix markup escaping fixes
add event argument to the bound function
merge pull request from sbilstein master
reapply styles specified in dropdowncss option when dropdown is opened to ensure they do not get remove fixes
fix more indentation issues
fix ajaxadapter s super constructor call
fix invalid index in buildchangedetails fixes
testing if the dropdown is opened before call positiondropdown on events by resize scroll and orientation
merge pull request from quantus master
remove event listener on destroy
accessibility fix role for nested options
merge pull request from alexanderzaytsev master
fix positioning with non static parents
corrected linting issue in turkish translation
more suitable spelling ещё instead of еще
resize search after remove a choice
fix dropdown items not being reset after filtering and selecting when closeonselect is false
add new loaded event fixes
fix bug in destroy method that would keep the original element hidden fixes
add missing letter
display error message on ajax errors
improve placeholder management fixes
merge pull request from agworld add load more padding option
merge pull request from bhoop master
dropdown repostion fix when previous postion changes container position
fix duplicate values in ie fix
merge pull request from bcoughlan master
fix selection first element from postprocessresults exclude selected elements from highlightable elements
add container to templateresult
perform deep merge for defaults set
add role and aria readonly attributes to single selection dropdown selected value
revert add title for choice item
add errorloading error message to sr sr latin and hr languages
add translation for errorloading
fix selection search resizesearch on close
escape special characters in containerid selector
switched back to this instead of self
fix tab index issues fixes
correct namespace issue
dont error out when x is remove really fast multiple times fixes
small cleanup in width regex matching
syntax error with ie embedded as activex in fat client
resize the search field to full width when placeholder is set fixes
pluralize items in formatselectiontoobig
when resolving width use case insensitive expression fixes
trigger focus event in focus function make chrome error
merge pull request from brankub master
method called using apply to avoid translating multiple args into single arg fixes
dropdowncss remove magic all correctly
fix tab key not respecting opts selectonblur
define term and context fixes
trivial change pluralize formatinputtooshort function
fix scroll caused by dropdown
add close event fixes
merge pull request from fusionbox master
updated indonesian translation
add accurate comment to the equal function
add the formatmatches string
allow ajax data to be undefined fixes
merge pull request from joshuajonah master
set maxlength on search when maximuminputlength is defined closes
use prop to get multiple attribute
prevent opening of a disabled select via api fixes
make search work with chinese japanese imes fixes
add missing finnish translations
merge pull request from kchung master
merge pull request from proloser master
move role and aria readonly attributes from render function to bind function
merge pull request from arendjr master
by default do not trigger change from val
fix typo in romanian
add the decorator parameter to the handlers for the attach and detach the dropdown otherwise the container is undefined
prevent form from being submitted when enter is pressed in multiselect fixes
updated malay translation
revert fix pluralization
merge pull request from hitosu master
fix make tabbing out work when no results message is shown
remove event bindings on destroy closes closes
fixes updated translation pattern
fix bug with non static parents for dropdowns
attempt to detect aborted requests
merge pull request from duckinator master
merge pull request from forby master
updated russian translation
remove dead code
call escape markup on results
allow markups of subsequent formatloadmore message
update defaults js
merge pull request from petermozesmerl master
lock firefox builds to linux
merge pull request from datapad reset filtered dropdown
add new options type parameter for ajax helper fixes
fix width and containercss incompatibility fixes
updated inputdata js
merge pull request from francoispluchino fix close focus
merge pull request from duncansmart clearing event
merge pull request from malekpour master
manage the more element slightly more efficiently
merge pull request from jrlynchii master
query as third parameter to ajax results callback
add computedstyle option for calculating the width
fix typo in maximum selected message
remove unnecessary code
updated sk js localisation
fix bug ie fires onbeforeunload event
fix line length to keep jslint happy
fix dropdown alignment in firefox still need to test ie
fix for random scrolling when using hierarchical data
fix bug where space was ignored in single selects fixes
pr for autofocus isn t supported
update base js
ignore modifier keys when enter is pressed fixes
ajax url function is now passed same arguments and called with same context as data option
no need to preload results they will be loaded on open
do not take over mouseup event in dropdown fixes
fix infinite recursion on focus fixes
useless space remove
fix problem with positioning the dropdown when it is on the right hand side of view
merge pull request from timisbusy master
goofed changed disabled to locked
add title for choice item
show search if minimum input length has been set
fix updateresults prevention if dropdown is closed
make tags matching case insensitive
fix selectonblur when closed by means other then tab key fixes
merge pull request from kevin brown master
fix french locale
preserve order of elements in the data call fixes
fixes removes redundant open event trigger
merge pull request from lenon fix maxlength
fixes better handling of ajax handler var
close on tab out fixes
add trim to opts createsearchchoice default function to avoid mismatches with val object
update vietnamese vi translation
better handling of empty string in select val issue
don t run sauce labs tests on pull requests
fix jshint errors due to too long lines on the attachbody adapter
recalculate dropdown height after drop above class add
merge pull request from bigmihail master
better handling of non numeric ids
fix selectioncssclass not working in defaults
improve percent width detection issue
remove dead code fixes
fix ajax data source error
set id in unescaped version
export classes issue
small bugfix in german translation
merge pull request from leipie master
only pick up direct children when retrieving data in multiple mode
merge pull request from kchung master
fix incorrect reference fixes
fix reference to undefined this triggering change on multiselect
fix missing placeholder closes
update focusing tests js
merge pull request from colemanw evaluatefix
issue blur on touchend seems more native
allow immediate typing when tabbing to multiple select
add support for data locked true next to locked locked attr fixes
copy title attribute from original element
merge pull request from eduardojmatos master
fix typo in comment
allow passing non jquery objects as dropdownparent
loading next page message formatting implemented so it could also have localization
update this context with data context on query calls
stopped refresh to top when closeonselect false
containercss remove magic all correctly fixes
add failing test for existing array selections
remove misplaced logging
merge pull request from datapad data result direct children
support selecting options with falsy values
fire change on clear fixes
corrected check for opening the dropdown
use click instead of mousedown so we dont break drag and drop support
remove duplicated options from multi valued select boxes
merge pull request from redbmk master
remove references to jquery events replace with the closure dollar variable
turn off autocorrect and autocapitilze
modified synccssclasses to use output of adapter when applying a class rather than applying the original class
missed one place where id is accessed issue
remove a unnecessary quote marks
handle null being returned from createtag
tweak dropdown position fixes
santize automatically generated ids
fix ctrl enter should not open
fix artifacts in pt pt translation
merge pull request from malekpour master
revert fix tab key not respecting opts selectonblur breaks tab key fixes
feed back the keypress code that was responsible for the close even
merge pull request from sambomartin master
ignore configured placeholder on selects without first empty option fixes
add missing french translation
merge pull request from manuel alvarez alvarez master
correct typo in pt js
merge pull request from ericbarnes tags tab
merge pull request from polonious master
cleaner way to detect older ie
disable ajax caching fixes
mark elements as selected when closeonselect is false so they cannot be selected again fixes
remove deprecated calls to jquery isfunction
optimize global mousemove listener
merge pull request from riatiger master
add inserttag to control the tag position
add zepto compatibiliy
use merged newdata in case where an option already exists when using arrayadapter
clear off active class when closing
add query to function head jsdoc block
formatmatches translation add
data method use the same triggerchange logic as val method
automatically initialize selection in tagging mode closes
allow overriding jquery s automatic cache busting on ajax calls
fix js error that happens when enter is pressed and there is no highlighted option fixes
render formatnomatches when all ajax results are selected
merge pull request from socialceramics url function with arguments
speed up calls to get the currently selected options
add third param to matcher in local
compress getcontainerwidth add missing semicolon
add missing errorloading to turkish translation
realign dropdown when search box moves to new line closes
update and improve persian fa translations
fix method to get space above the dropdown
merge pull request from yallups terminating script undefined args
clear placeholder before opening dropdown fixes
remove experimental code that accidentally got checked in
merge pull request from jorupp master
tab to spaces
issue prevent delayed focus when the list is hidden
add tab support for tags
update focusing tests js
prevent the cursor from setting it to initial position in ie
fix a typo in the docs
fix initselection with empty input value
merge pull request from diogotrentini master
fix keyboard removal of choices fixes
prevent scrolling when dropdown is opened fixes
strip bom from encoding closes
merge pull request from arthur peka master
allow passing a function for maximumselectionsize
forgot to remove log
french locale correct file encoding and diacritics
fix for some selects not closing fixes
add a little animation for remove multiselect choices
merge pull request from polonious master
merge pull request from ebraminio master
a small optimization to how many times highlight is called
merge pull request from mxab master
merge pull request from ryfeng master
renamed this var
check if search should be focused when the dropdown is activated
tweak formatselection for
merge pull request from hoopyfrood master
fix chrome issue lost focus after selecting a value
merge pull request from damienfinck master
automatically push docs updates
dont open the dropdown on meta keys fixes
possible fix for issue better blur support for touch devices like ipad
optional search input placeholder parameter add
safer check for result children
merge pull request from lazyhammer fix choice focus
allow formatselection to return a jquery object
add failing tests
fixes cutoff for bottom option when selecting with keyboard
copy style attr from original element fixes
bugfix for resetting scrollpoint on multi select
workaround for chrome focus bug fixes
update es js
set all ajax parameters as query string parameters
fix existing selections not being respected with array data
merge pull request from thedistantsea feature remove hardcoded classnames
attach the dropdown to the element s body instead of the current body issue
update base js
updated sk js localisation
merge pull request from ycdtosa master
merge pull request from alexryd remove bom
change condition to use length
merge pull request from junhanamaki master
add loadmorepadding option
formatmatches translation add
merge pull request from jhinch pixel rounding problems
do not close on select if ctrl or meta cmd keys are being held as described in
append label only when formatted is not undefined
fix internal function indexof when undefined passed
fix js breakage due to undefined variable viewportleft
firefox tab and enter keys issue
fix plural suffix for polish translation
always set the val on element so initselection has access to it fixes
merge pull request from murb lt file jshint fix
fix dropdownautowidth so it actually works
fix da language apply
improve french translation
fix form focus lost
fix typos in test
fix long line for ci check
clicking the clear button raised the changed event twice fixes
allow a custom function for computing the width fixes
focus is lost when the change event is triggered
add an integration test to ensure issue does not come back in the future
expand ie testing
implement dropdown bind
formatsearching can be disabled
updated language name on comments
fix placeholder missing on tab out on input tags
floor the searchwidth calculation due to inconsistencies between fractional pixel handling in different browsers
improve highlighting in single selects
remove shrinking animation on choice remove because it causes jitter fixes
merge pull request from ebbe brandstrup master
repack hebrew translation closes
trigger the select event in the tokenizer
merge pull request from zapodot master
merge pull request from moneybird master
merge pull request from niquola master
do not open when clicked on a search choice fixes
add tests for case insensitive tag matching
avoid error when label has no matching element
save data values to correct context
fix for uncaught typeerror
improve performance on dropdown detaching
merge pull request from kris lab master
add support for placeholder attribute in addition to data placeholder
merge pull request from datadog master
clarify syntax fixes
pass along arguments sent to debounce
mange disabled attr on source element fixes
merge pull request from ifours master
avoid locked options to be selected and deleted
minor fix greek omega used has no diacritic
merge pull request from skivvies master
exposed open close as public issue
merge pull request from wdietz master
merge pull request from flyerhzm append label only when formatted defined
merge pull request from michalzalewski master
fix anchor hrefs
make keyup change event work with repeated backspace delete keys fixes closes
pull requests are marked as false not missing
reevaluate dropdowncss when positioning dropdown
fix markup escaping fixes
fix bug with determing the width fixes
give the sizer a higher offset just in case its on a highres screen
merge pull request from christophano master
fix minor errors e g redefined variable
fix typo fixes
swap inputtoolong and inputtooshort error messages
fix focus in mobile fixes
fix focus issue on multi value select
merge pull request from norlin master
updated indonesian translation
merge pull request from hzy master
add missing norwegian bokmaal label used for errors
no need for style if we already have a sizer
remove the byte order mark
fix typo in error message
fix incorrect classes on loading more results message
merge pull request from pmirshad master
fix width and containercss incompatibility fixes
width copy option needs to ignore min width and max width fixes
fixes and issue where if args are not present script terminates
attr to prop change
add todo for placeholder validation
remove duplication of no matches found message
minor formatting cleanup
allow clearing on backspace as well as delete fixes
increase statuscheckattempts in saucelabs qunit tests
corrected danish translation for results errors
improve focus handling on mobiles fixes
create dsb js
check for string value
also check if search input should be focussed for multiselects
don t rely on visible and not selectors which only work with sizzle
fix lithuanian translation
restoring allowedmethod container
correctly filter out selected options out of elements loaded by infinite scroll fixes
extra last resort
simplified norwegian nb translation to a more correct text
correct reference for this in ajax callbacks
fix opening and closing of dropdown when an element is remove from selection fixes
better handling of tags that are not just strings fixes
consistent containid for better integration with form validation
fix autocaptilize typo fixes
clear search field when data is used
fix jshint warnings
reuse result of previous calcualtion
remove debugging statement
merge pull request from jdecuyper feature remove duplicated option
fix focus on select
explicitly require jquery mousewheel now
recursively apply defaults
fix the fix for
add errorloading translation of thai language
store last mouse position in a variable instead of document datastore
update french translation
on clear cleared tooltip too when placeholder present
original option not available to formatselection on init
fix the title attribute for choices on multiple widgets
clear the placeholder on open
avoid call positiondropdown function if the container it s closed
remove debug statement
label for support for id s with brackets eg id name first fixes
remove trailing comma fixes
improve keyup change detection
multiselect javascript error no method showsearch fixes
renamed to search
allow an empty string to be used as the placeholder title
deal with non existent style attributes and returning calculated pixel widths as a fallback
fix secure fix to prevent access to undefined array index
fix line length issue
fix testing mistake
update populate at formatresults to use bulk append of nodes
rename tagseparator to separator
fix there their typo
merge pull request from tpruvot master
reverted a line in base js regarding tab index
bugfix movehighlight disabled list elements were highlighted now they are ignored
fix hiding of search when minimumresultsforsearch are used fixes
merge pull request from maxxsoftware master
fix not working adaptdropdowncssclass callback
update de js
clean up some comments
merge pull request from mkurz focus fix
merge pull request from backstop master
merge pull request from jibwa master
after triggering change also trigger click to help some validation frameworks closes closes
style detection for search field sizer corrected
fix typing mistake fasle instead of false
relax formatsearching check
improve resizesearch performance
fix focus issue when select input has placeholder
allow to pass a callback as ajax url
set platform for all browsers in grunt saucelabs tests
ie and compat mode
fixes error when getting offset of element not in document
recalculation dropdown height if dropdownautowidth is true
update sk js
fixes remove some dead code
create hsb js
add the right single quotation mark to diactritics
do not open on meta key closes
try the testtimeout param instead
pass element attr to initselection for selects upon init
fix deep cloning options
better extract width style from inline element
only disable keyboard focusing for touch devices fixes
merge pull request from socialceramics postdestroy dropdown error
on clear cleared tooltip too when placeholder present
merge pull request from streid master
merge pull request from dgroux master
fix remove an hard coded title
improve whitespace detection
remove add line
aborts the ajax call when a new is coming up
merge pull request from avelis master
replace this element val with this current
use more specific selector for finding options of a select
slash doesn t escaped
fix nofallthroughcasesinswitch jsx object is not extensible
set the public path to the asset manifest entries
revert change no unused vars args from none to all to show warning on destructured objects
fix checking against non semver compliant yarn versions
add wait false when opening app in browser
update commit message to use imperative mood
support jetbrains rider ide as an editor
add an unobtrusive log when server disconnects
small typo fix
add link to deployment docs after build
remove flowtype require valid file annotation temporarily
a bit of lint relaxing
recommend st instead of http server
adjust typescript eslint to not warn about typedefs when used before defined
use posix paths for jest config during eject
use a more solid way to reset the console
exit with an error code when npm yarn install fails
fix click to open on windows
make build exit with error code when interrupted
fix typo in comment is right not
fix project cleanup on windows
feat remove unused react imports
remove unnecessary describe
improve reliability of port hint
revert set the public path to the asset manifest entries
add code insiders to the editor list
fix typo in createreactapp js warning message
refactor remove double coerce
fix default package version add by global cli
clear scrollback in test mode
make compiler variable const instead of let
exclude d ts files from typescript glob
add simplified service worker invalidation
use process execpath to spawn node subprocess
disable react no deprecated rule
only use safe options when packing css assets
use yarn in predeploy script if built using yarn
add current version and bin location to info output
add additional information for postcss errors
tweak init js after merge
feed correct line number to gvim on launch
simplify protocol https true false to protocol https
pass jsx runtime setting to babel preset in jest config
fix floating caret position incorrect while scrolling overlay
make applescript call more bulletproof and silence its output
webpackhotdev now uses wss when https is used
shame on me
add missing dev argument
add eject survey
add missing n to the end of package json file
don t strip stack traces of evaluated webpack bundles
revert check for multiple project names when initializing
output css sourcemaps in separate file
promote react must be in scope to be an error
treat css and sass scss as side effectful
fix the code splitting links in the eslint output
update the thread loader test in errors
merge pull request from lacker cleanlint
fix windows compatibility
print full directory name from lsof
support pycharm in launcheditor
check internet connectivity with lookup instead of resolve
add feature extended jest config
disable copy to clipboard in cra info
unmount the app in the default test
only show first error
enable node environment for eslint
fix dead link
support passing scoped package names to scripts version arg
further tweak runtime plugin per documentation
tweak scroll sequences
fix tsconfig json lib suggested value
ensure proxy url starts with http or https
fix the order of arguments in spawned child proc
add missing period
type check json files
fix a typo
remove the at the end of the init message
add babel sourcemap to resolve vscode debugging issues
fix code comment typo
fall back to displaying full error if message is missing
respect process env port
remove unneeded url lib
open url in chrome if app is already opened
fix react jsx error
remove interactive shell check when opening browser on start
improve vendor chunk names in development
promote no undef to be treated as an error
add empty mock for dns
rename getting started link to docs
if template name already starts with cra template leave it alone
disable react constant elements because of bugs
ignore intellij module files when generating an app
change cssmodule classname hash to use repo relative paths
edit interpolatehtmlplugin hook
deactivate the new cap rule in eslint
switch to eval source map
add useful link to react scripts
add warning when using react error overlay in production
add namedmodulesplugin in webpack config dev js
support configuring coveragepathignorepatterns
revert lint related changes made in
include pathinfo in dev
disable comparisons feature in uglify compression in production
add testurl to jest config
suggest just yarn build
return proper exit code in bin react scripts via spawnsync
ensure address is private
use file name whitelist to prevent rce
fix sass importloaders
don t look for babelrc file during test
move the remove on eject end tag at the end of the file
simplify path resolving for react scripts
remove api reference from footer
strip comments from the production build
fix module function name in error overlay
webpack dev server patch for still ok success status
increase workbox s maximumfilesizetocacheinbytes
ux explain why build is failing
set resetmocks to true by default in jest config
inlinechunkhtmlplugin works with empty publicpath
add check to filter out custom scripts in eject command
use wmic to get process list on windows
relaxed eslint rule no unused expressions
fixes unchecked access to deploy script on build
start the dev server at the specified host
mjs should not resolve before js files
fix file size reporter on windows
export dismissruntimeerrors function
proxy rewrites origin header to match the target server url
jsx no duplicate props ignore case fix
update docusaurus config
add helpful message to the footer
make gh pages support more discoverable
fixup gh pages deploy instructions
only suggest that tsconfig json is incorrect when syntaxerror is caught
i think the word listening is less clear
adjust regex fed to regexp
revert switch to eval source map
move unused eslint webpack import into remove on eject block
enable eslint caching in development
account for browserslist environment variables
shut down server silently as it is too noisy
add a missing color mapping
internal webpack paths should not be clickable
add link to better visualizing tool
setting a dynamic port value for the pushstate server url text
fix script name to open chrome
revert add support for new yarn workspaces config format
fix resolve eslint config from apppath
add slashes to websocket protocol url
docs add alt tag to homepage logo
fix typo and be clearer about ecma settings in uglifyjs options
work around weird inferred name in chrome
add eslintconfig to new projects automatically
allow declaring variables before use in a scope above
small grammatical correction
make sass missing message friendlier
enable forceconsistentcasinginfilenames by default
don t auto exclude test files
add a localhost only log message pointing folks to the pwa docs
mention npm test in init output
ensure workspaces key exists in parent package json
improve error messaging in verifypackagetree js
check for multiple project names when initializing
exclude html from default loader
openbrowser causes docker compose set ups to fail because of spawn eaccess
extend scripts version to include tar gz format
add support for new yarn workspaces config format
remove accidental exclamation
remove extra eslint rules
docs babel preset update comment info about babel preset env
remove bin files defined at react scripts package json after eject
replace text align start with left to fix edge
fix unlogged yarn pnp message
remove project property from typescript eslint parser options
fix a potential issue
enable compression on webpack dev server
suggest yarn build rather than yarn run build
allow use of scoped packages with a pinned version
remove chalk from formatwebpackmessages
add empty mock for module
allowtaggedtemplates to avoid warnings from sc s
fixes port env variable not always an integer
show customized build path for serve instruction
fix css font face minification
move browsers to cross tool config
remove guard for in lint rule
remove duplicate url key in docusaurus siteconfig
do not check for interactive session to shut down dev server
normalize babel caching across the board
specify roots in jest config
support globalsetup and globalteardown jest options
instruct jest to load native components from rnw instead of rn
avoid typeerror on when registering service workers without config
update short link for pwa
update eject message
add restoremocks to supported jest config keys
change app component declaration from arrow function to regular function
relax label rules
remove options in uglifyjsplugin which are equal to the defaults
configure which browser to open in npm start
provide empty node mocks for fs net tls
check if ci variable is set to false
update the string that clears the console
add x forwarded headers for proxy requests
fix eject on windows
reorder vim arguments in launcheditor so remote works
eliminate regenerator from preset react app plugins
add dynamic import transformer for dependencies in test env
fix absolute paths issue in jest
fix react is not defined compilation error after ejected
ignore the yarn cache directory
fix shortcut handling in react error overlay
add permissive ts lib defaults
improve eject message
relax no cond assign rule
minify and map css
enable hot reloading for css
disable jest watch mode when coverage flag is present
fix broken css module support in prod
use custom bit ly links
disable second pass of autoprefixer
fix typescript file detection
make jsx no undef rule an error
disabled ts no undef rule
fix minimum version error on old versions of node js
fix the build
merge pull request from maciej ka remove trailing spaces from copyright
sync dependencies babel config
disable no dupe class members for typescript
clarify why we use find cache dir
remove no return assign lint rule
catch and noop call to open web browser
add brackets to launcheditor js
feat react scripts allow testmatch for jest config
remove radix eslint rule
convert json stringify n to os eol when writing tsconfig json
change browsers output to be a single line
fix cli grammar
closes webpack dev server and exits process on end stdin
reference issue to test
add some stuff that requires transpilation
ensureslash fix accidental string to nan coercion
use file name whitelist to prevent rce
use modern syntax feature
skip gh page setup instruction if scripts deploy has been add
chore fix broken link for cra deployment
do not parse eslintignore
remove stripcolor usage
add a reasonable config for autoprefixer resolves
change create react app version in react scripts eject warning
changes no unused expressions lint from warning to error
add explicit check for watchall false
fix build deployment instruction link appends comma at end
map css sass scss modules to identity obj proxy in jest
remove build directory without ending up in trash
updates comment to reflect codebase
fix npm test on windows
merge pull request from goncy unused vars not showing on destructured objects
add thread loader config for application code entrypoint
remove punctuation for consistency
gracefully shut down the server on signals
resolve app paths from realpath of cwd fix
ensure dependencies exist in package json
fix unfound export error
remove trailing spaces from copyright
unmapper windows compatibility
remove stats json
properly exclude test files from typechecking
fix comment typo
don t resolve localhost if on network
fix regexp from navigatefallbackblacklist workbox
revert ensure workspaces key exists in parent package json
fix parsing html jsx tags to real elements
accept version when loading fonts e g font awesome
fix a bad condition
skip stdin resuming to support lerna parallel
prioritize index css over the implicit app css
enable css sourcemaps in production
add babel runtime version to transform runtime plugin to reduce
replace template literal fixes
add editor support for sublime dev vscode insiders
autodetect macvim editor
remove coverage watch workaround
fix windows build
false expression should not be in dependencies
disallow event global variable with no restricted globals rule
use a better clear console sequence
minor tweaks for crash overlay
send proper error response in custom onerror handler for httpproxymiddleware
support phpstorm in launcheditor
update index js
check for more packages which may cause problems
fix seperators typo
remove outdated comment
override no unused expressions with the typescript eslint version
minor performance improvements
disable jsx handler name eslint rule
focus on line in eslint messages
exempt variables prefixed with underscore from no unused vars rule
clean formatmessage usage
put console stack behind a react vendor prefix
turn off operator assignment stylistic rule
add edit feature to docusaurus pages
add regression test for
fix up broken line
suggest jsx react in tsconfig
tweak initial output
remove dot from the end of the link
fix eslint complaints in build js
remove other flow command
order docs in footer
fix the right and bottom padding of the error overlay
use workbox cdn as default
fix typo in code comment
fix order of args in tasks cra test
set output globalobject to this
fix hmr in firefox when proxy option present
update gh deploy instructions to match user guide
turn off babelrc usage
add algolia search bar to docusaurus
bail for production builds
enable eslintignore with extend flag
change no unused vars args from none to all to show warning on destructured objects
whitelist main in template json
allow additional jest config keys
remove output path from dev webpack config
adjust browser defaults
add vscodium to the editor list
display pid in already running message
add compile progress indication
configured the thread loader to keeping workers alive in development mode
remove unused eslint comment
remove superfluous lodash usage
fix variable name casing in filetransform
use const instead of var in env js
set npm loglevel to error
always lint with latest react version
corrected typo in test
add separate render test
made changes based on jslint
fix test about curve
small gramatical mistake
remove logging from sequence diagram test
make gitgraph commit ids out of hexadecimal chars
test ganttdb spec add simple tests for the durationtodate fn
fix foreignobject issue
revert to http
remove escape inside regex for standardjs
fix for when the br tag is used
make testing quieter
fix for issue with bindfunctions being called when there are none
update config js
gantt callback default to node id as callback arg
adjusting initial classdiagram shape rendering
merge pull request from soptq staterenderer
fix the binding function error
fix lint issue
merge pull request from daveaglick master
in render add calling callback for gantt functions
fix sequence diagram node text alignment
run chrome with no sandbox
add viewbox attr to class diagram
make default node number a little smaller
small fixes for linting
merge pull request from vaniship master
make default sequence numbers a little smaller
merge pull request from raghur master
fix fix draw function can only call once
fix notes for state diagrams beta
fix git graph not clearing
merge pull request from progressive insurance bug git graph clear
add utilty function to check for substring in array
fix a typo crosshead arrowhead
update phantomscript js
test ganttdb spec clear function
fix logger to display exception stack includig message not just
add unit tests for issubstringinarray utility function
jison token stack
for multi line labels separated with n
remove use strict
add init argument to the global api
fix if theme inclusion for classdiagrams
add test for multiple class diagram
fix class diagram
merge pull request from ma zal master
addressing code style issues
fix for link styling and classes for subgraph
fix logging output in prettyprint
fix render multiple relations
fix regex and fix incorrect test
resolved an name change issue fom a merge
cli propagate exit code from lib i e phantomjs
ie local html cssrules access is denied
merge pull request from tylerlong master
address code style issues
fix sequencediagram over note horizontal position
fix for broken test cases
merge of pr
fix for build issue lint
fix an incorrect reference to an outer variable actor when processing actor widths
add test for using a default style
make path a bit wider
infer svg height width from of nodes
merge pull request
fix exclude text with line breaks when parsing wrap setting
flowrender js fix fontawesome icon insert
handling case with concurrency lines
fix for init not running by default
test pass with underscore in a class name
remove a tag moved id attribute to text section
add conditional to add fill none to styles array if fill not present
refactor ganttdb separate out the inclusive end date toggle
fix dotted lines not appearing in flowcharts when html labels disabled
fix reference in test file
testing the label width in percy
allow other forms of node selection for init
fix text label colors for flow charts
other typos in flowdb
fix code style issue
test ganttdb spec flip function design to specify inclusive instead
fix gantt chart cli configuration parsing including functions
narrow down console logging when flowchart subgraphs are add
fix issue which only passed the first char of functionargs to the function by splitting the functionargs by commas
fix a flowchat selection issue
add usemaxwidth to staterenderer
use chrome stable instead of chrome canary
missing call keyword
fix for broken flowchart test cases
fix relation got cut off
use a library level variable for assigning ids
zoom in a bit
use timestamp to generate uniq id
code style fix
allow node circle to receive css styles
fix for merge error
better margin calculation when there are several diagrams on one page
remove whitespace to adhere to standards
add cypress test case for class diagram cardinality support
minor change to code
renumber render tests
merge pull request from gnkm add viewbox to class diagram
merge pull request from ivan danilov sequence over notes position fix
add fix another tests
allow generics support for class title definition
fix a typo
corrected the test case description
fix for issue add unix new line at the end of generated svg
update config js
fix api gantt s side paddings
test link target
bug fix handling non click graphs
remove duplicate code
fix for broken build issue with config use in flowcahart and gantt diagram
fix of broken logger
cannot center justify text in nodes
support latest version of font awesome
fix for tests
fix for test
add case block for mermaid api parser
test ganttdb update inclusive end date tests
fix a typo
merge remote tracking branch origin master
output all cli log messages in the same way
fix for broken build
merge pull request from arthaey remove console log
fix calculate text dimensions after wrapping message text
some tweaks for sequence diagram
fix for legacy class diagram title for class bold styling
add comments about css in config
fall back to using the default mermaid style
add rendering tests
fix for test
another style fix
usage of path join for add styles as per suggestions from fardog
missed activationwidth new configuration option for sequence diagrams
multiline sequence message
remove extra space
flowchart add class clickable on vertices with link or click function
small fix for import
remoed a console log
merge pull request from tylerlong master
fix for dashed line dependency support for classdiagram
fix line styling dashed thick when not using html labels option
refactor css loading code
tweak for lint
adjusted integration test
changed tspan as default of sequence actor text placement
small for for merge issue
fix for failing yarn dev on standard syntax check for mermaidapi js
dark theme fixes for gantt
test ganttdb spec fix typo
add jest test for dash line dependecy parsing
merge pull request from wfnuser fix typos in flowdb
fix width of notes left of and right of
update theme base js
path fix for jison parser generation
fix remove it only
fix for issue
merge pull request from bfriedz circle styles
fix typo pats paths
modified the number of week for which the x axis stays in week mode as discussed in issue
merge pull request from stanhu sh fix dotted lines flowchart
update mermaidapi js
test ganttdb spec fix missing initialization for clear test
merge pull request from larryklugerds master
remove useless whitespace
remove console log in classdb
better align activation box start
fix git graph
add another test for checkout
simplify inline css code
fix for test
fix note text y position
fix stylesheets index issue
fix a new config issue
add cypress test for dash line dependecy rendering
fix fr background in notes when using htmllabels
make logger print object instead of string concatentation
uses an empty text node instead of a string for svg group labels
updated variable name to cnf
merge pull request from ileler develop
link or click must both work
fix ganttdb js clear formats and excludes on clear
test gantt spec add parser test for inclusive end dates
fix window is not defined with ssr
reenabled all tests
use isplainobject instead of isobject
add exception for innerref styledcomponent render
ignore one more flow issue
merged the autoprefixer changes
fix variable name
fix rollup plugin babel warning
suppress a warning in a test to reduce noise
refactor generatealphabeticname from recursive to iterative
some small linting fixes
exclude priority bumps from replacement
update the test description
fix servertag removerules not emptying marker correctly
adapt to style rules
revisions to existing elements must go in a new tag for streaming
refactor code to make it consistent with styledcomponent
suppressing flow error on tests
set styledcomponent context directly on every class
remove a dev warning from production build
refactor generatealphabeticname for shrinking bundlesize
support non simple objects
fix typing of themeprovider
fix createglobalstyle typo in the comment
hammer onref onstyledref
hammer callback reuse
fix unit test again
allow resetting style cache
fix stylis usage
expose version in api
fix theming for react native
remove browser code from server bundles
use shared empty in a few more places
shave off some unnecessary bytes
fix themeprovider hmr
fix flow type annotation
replace murmurhash implementation
fix whitespace test errors
dont import all of lodash
remove that pesky preceding space from the data attr
fix more tests
fix test noise
same deal for native primitives
make usetheme cleaner
merge pull request from styled components css variables support
any references to path break react native
simplifying the too many classes warning
merge pull request from styled components fix build
update error message for invalid targets
error message support for elements and components if used to interpolate incorrectly
fix add unit test for createglobalstyle when style tag is detached from dom and then remove the rules
stripping the new comments out of the test helper
fix tests on vendor prefixes for display
clean up constants test js
not sure if theres any reason not to do this
cache themeprovider inner function
merge pull request from vepor css without interpolations
remove unnecessary props check in state constructor
fired up the example project and got at least the simple case working
fix innerref on html elements
ignoring this error for now
check for the selectors array to have a result
fix broken errors md link
improve consolidatestreamedstyles warning message
changed to glamorstyle
flow is pretty k tho
change module detection to prevent multiple warnings
rename heridity with inheritance
remove console error for insertrule
this test isn t really relevant any more
merge pull request from shannonrothe master
change type to boolean
skip this test for now
tweak rehydration output
add a test around css rule order
add documentation comment
add onreset to valid react props list
fix src models test stylesheet test js to use injectglobal factory
remove calc test
revise simplify theme determination
dont inherit displayname or componentid
fix styledcomponent and stylednativecomponent constructor
fix deferred arrays being mutated while used across clones
add failing test for media at rule inside style rule
remove unused lifecycle methods
update postinstall wording
skip attrs if it s undefined
fix prettier error
chore fix danger logic for if tests are included with lib changes
reorder things for better minification
fix styletags typo in comments
fix benchmarks build
reorganize types to eliminate cycles
refactor sandbox example buttons to especially improve mobile appearance
add annotations to expectcssmatches
skip multi instance warning in test environments
add warning if there s c was initialized once on the page
adjust when the cgs multicompose warning is fired
add the button of react native entry point
faster version of interleave
change attrs property to attrs method in warning
use strict equality
update styled js
update themeprovider test js
fix title style
adjust minification mechanism
one more dev error
add test for react memo hoc
extract regex from styledcomponent scope
order is important
fix injectglobal bug
add ua type check
remove unused flow surpression in no parser test
fix typo in createwarntoomanyclasses
add failing test for nested media queries
commented where this came from
add failing simple test
add new error system
remove stylis rule splitter from stringifyrules
get rid of the themeprovider single child context restriction
not passing on non html attributes
remove unactionable danger message
revert add tests for font face support
suppress warnings for stylesheetmanager
add displayname to returned component
switch startswith to normal regex
fix type errors
merge pull request from strozw fix sandbox undefined error
merge pull request from cadaverine fix typo
run all tests again
skip the current stylesheet tests for now
convert object assign to spread operator in tests
add warning when using keyframes on react native
clean up merge conflict
remove secret internals from react native unneeded
dont output empty component tags in production
always inject an empty object as props theme
fix resetstyled test utility
move the newline out of the css tests still pass
omit some development errors from production build
fix typo in failing test case
this test just passes amazing
dangerfile touche up
add proptypes and default props for cgs suppressmultimountwarning
fix unnecessary test failures
refactoring broke out a tag class
fix whitespace related test failures
disable webpackdevserver logging
only collect coverage for pull requests
suppress false positive flow errors
drop the space at the beginning of the class names
remove unused composition
merge pull request from styled components keyframes native warning
add failing test to attribute filtering
remove deprecated attrs suggestion from too many classes warning
merge pull request from styled components adjust cgs concurrent warning
remove empty object as argument
add failing test
remove dependencies from es bundle
also add the util to the other indexes
fix duplication detection
fix models styletags nits
failing test withcomponent attrs is not a function
merge pull request from styled components handle hmr dispose cgs
simplify object spread
use babel helper module imports to import modules
change usage of wrapper node to wrapper html where it gives same test result
add deprecation warning to no parser bundle
readd guard around componentstyle deferred injection
better error message for non styled component interpolation
remove donate from opencollective link
add allow as a valid attribute
fix styling order
remove innerref when it shouldn t be set
still just tweaking this api
merge pull request from didierfranc master
fix styledcomponent displayname is displayed as styled undefined when using stateless component using component name when component displayname is undefined
add marquee tag to default attributes list
oh replacing is not what we want we want appending but at a particular point
tidy up logs
fix standalone build
ignore linter barking
speed up hyphenating style names
avoid allocation in componentstyle
add comment explaining guard for edited this el references
fix new stylis api
rename generatename in componentstyle
add short url to warning
attach displayname to forwardref function as described in react docs
fix rollup build failing
fix typo for generatedstyled in stylednativecomponent
remove injectglobal deprecation warning
remove legacy react attributes
add triple inheritance test
merge pull request from manpenaloza better sandbox stylings
remove extra object spread
get rid of the themeprovider single child context restriction
remove now unused type
fix stylesheet test
start a whitelist for props like this
fix makespeedytag s css method
fix injectglobal deprecation not being guarded
add next and all sibling selector nesting
stronger type check for object refs
bug fix innerref value for nested styled components
update inheritance in stylenativecomponent js
ohh yeahh define and inherit can work
fix incorrect element output being created by styletags
fix unit test
add typeof process check
propagate fix for threepointone glamor
fix keyframe test
add tests for font face support
fix generatealphabeticname with negative numbers
fix undefined warnextenddeprecated call in production mode
update isstyledcomponent now that forwardref is being used
fix interpolation annotation
insert extend in withcomponenet attrs test
make a generateandinjectstyles function in stylednativecomponents
rm unneeded api exports
bug restore setnativeprops
using loose mode for transform class properties
avoid destructuring tagged template function arguments
add one more case
remove usage of typeof process
add flow annotation in utils
fix module resolve error
updates global styles nonce test to include valid style example
fix withtheme reactclass type
fix groupedtag buffer resizing
fix createglobalstyle injectglobal
fix typo copmonent component
merge pull request from styled components fix tests
fix console error in attrs test
pass props to interpolations in stylednativecomponents
manually merged in the flow fixes from master
merge pull request from styled components fix comp constructor
make styledcomponent inherit parentcomponent s contexttypes
better typing for emotion unitless
switch globalstylecomponent to a purecomponent
add imagebackground to native aliases
avoid an allocation
add knowledge of react is typeof to rollup
add test to codify cgs injection order as part of the api
remove unnecessary global property type from serverstylesheet
fix cgs determinetheme usage
add protocol to open collective link
add todo comment for implementing css escape eventually
simplify styled constructor
only delete innerref if it is being passed down to html element
get rid of for of loops
fix rollup s stream warning by moving require down
forward innerref prop
update attrs test js
ensure edited textnode is appended to this el
typing determinetheme props parameter for theme attribute
use createtextnode instead of textcontent
replace process env only in prod
change typeof check
good to have you back danger
prevent cgs styles from being lost in an out of order render unmount cycle
remove unused test code
merge pull request from styled components fix edge case
nix some allocations rename variables to be correct
proper fallback for stylednativecomponent
fix test now that the path changed
merge pull request from maciej ka fix unknown prop console error in test
fix findlaststyletag for text nodes
fix extending styled component
remove an allocation
remove flow suppresion
remove obsolete tests
fix classname being overwritten in styledcomponent
fix styledcomponent and stylednativecomponent constructor
add tests to withtheme hoc
fix flow error
unnecessary flattening and interleave of css without interpolations
refactored constructing babel config in rollup config
merge pull request from styled components fix unclickable opencollective link
update extending test case for stylis rule merging
make sure theme passed in with props overrides themeprovider
add flow comment to the files missing it
moves the nonce attribute attachement code outside of the name check
the warning too many rules needs this low level api
add source maps to the client build
pass defaultprops from parent to child styled component
add innerref support to withtheme hoc
remove extra semi
console log and capitalize
merge pull request from styled components perf
fix splitbyrules using cascade false with stylis
provent css flattening until it s required
remove console log
test use node env for ssr tests adjust test util to handle that
defer injection until first render so it appears after anything loaded with css loader
add enzymefind test util
adjust rehydration regex to handle deeply nested mqs
ugh flow and eslint conflicting
remove array from use from browserstylesheet
tone down dev warning for deprecated injectglobal
add smart default for getcss
don t export primitives
broke this when i forked postcss nested
change navigator check to prevent crashing in node env
fix wording src packages styled components base js
forgot to replace n to in classname
fixup fix fix duplicate slash in metadata url
ci bump tslint timeout
chore handle package scripts being empty when generating readme
fix prevent snoretoast shortcut set notif title
fix detectlanguage function
fix ts ci
fix missing comma
fix plugin options can be missing when rungenerator is directly called
fix fix absolute path for outputdir option
fix should respect root parallel option in terser plugin
fix serve don t set header origin if using an agent
fix ui remove console log
docs add manifest in html head
fix fix cli version check cache
chore pre release sync
refactor changed var async to isasync
feat babel preset app pass full config to babel preset env
fix cli service pass public host to devserver
chore pre release sync
add log for completion hooks
workflow ensure syncs are committed before release
fix add no verify to initial git commit
workflow run checklink sequentially
chore fix typo
fix ui add router button should not require prompt in terminal
fixup chore deprecate tslint
fix ui prompts choices values
bump globalservice timeout
fix should add shamefully flatten for pnpm install
test bump test timeout
workflow allow manually specifying dist tag when releasing
fix show full project name in error message for vue create
fix adjust postcss loader order when using inline minification
fix ui fix the incorrect regexp used for cors check
chore pre release sync
workflow fix pre release sync errors
fix plugin api cwd is not a function
chore pre release sync
fix migrator correctly extract config fields to files
feat allow disabling serve progress via devserver progress
ci fix invoke spec
chore pre release sync
fix ui bail on error reading package json closes
fix airbnb eslint config
test further bump pwa wait time on ci
chore fix the name of the tested file
increase eslint test timeout
fix ui create first feature remove if use config files not checked
fix eslint migrator fix local eslint major version detection
fix fix hmr compatibility with worker loader
fix ui client addon serve error
fix nightwatch compat with legacy presets without webdrivers field
fix ensure paths make html optional
fix fix eslint errors when using airbnb config
make sure to await writefiletree
fix sortobject should not introduce formerly non existent fields
fix cli service process the webpack failed hook in the serve command
feat add no fix warnings option
fix build respect productionsourcemap option for all targets
workflow auto commit changelog on release
ci bump cypress test timeout
fix typescript fix ts modern mode
fix pwa set cacheid in generatesw mode only
ci fix index html moving
test add test for
chore pre release sync
improve test file format
fix jest make sure jest tests work without babel
feat set whitespace condense for template compiler
chore add module concatenation hint
feat wc entry accepts multiple file patterns splited by
feat serve allow specifying public network url via command line
chore bump nightwatch test timeout
test skip the eslint migrator test that constantly times out for now
fix cli avoid assertion error when vue config js includes assignment expression
build update syncdeps script
test fix failing tests that due to async support in generator
feat enable splitchunks by default even in development mode
fix enable html doctype by default for pug
feat allow specifying proxy when creating project
fix ui eslint disaled rule
fix fix injectimports when there s no pre existing import declarations
fix ui config file
fix ui new update available message adapted
refactor use createrequire to load resolve modules
fix fix jest test match
refactor use inline approach
ci use system chromedriver on appveyor
fix jest should only run files in given directory
fix should not throw when a plugin listed in optionaldependencies is not installed
feat generate npmrc when using pnpm as package manager
feat vue config devbaseurl fix
fix ui client addon config
fix resolvepluginid should correctly resolve vue cli service
fix generate readme with code blocks
ci fix test
feat allow specifying additional configs in preset
chore skip confirmifgitdirty in api mode for now
fix cli ui ignore false enoent errors on windows
fix jest typeahead config
test fix typescript test
chore lint files
chore fix typo babel
fix typescript fix typo in lint command registration
chore ui display error if plugin loading fail in debug mode
sync open in editor url w devtool
feat service allow mocha unit tests debugger to be bound to a specified ip and port
fix refresh page switching between views doesn t lose selected item
fix fix modern mode in monorepo by resolving in the package scope
fix clear children modules from cache
fix ui remove tasks not being properly filtered
workflow don t sync deps for now
workflow fix vue cli version marker version bump for prereleases
chore fix prompt message
fix externalize vue server renderer support dynamic import in mocha tests
docs fix typo ci skip
feat preservewhitespace false
fix invoke not reading dot files
docs remove run for yarn commands in readme
docs fix comment typo
chore move add before invoke in commands
fix fix vue cli service initial version
fix correctly sort test scripts in package json
fix fix require vue babel preset app version return undefined bug
test fix unhandled promise
test remove only
fix ui check current project still exists
fix comment eslint disable in cypress config
chore remove extraneous log
fix fix core js import for global service
fix respect dotfiles in public dir fix
refactor ui views now stored per project
feat serve support entry in vue cli service serve
fix use the correct size unit kib
fix ui ipc send
fix ui hasplugin crash
fix airbnb lint should not warn on vuex state mutation
fix vuepress docs branch
fix typescript ensure ts loader options can be stringified for thread loader
fix inspect correct usage of resolve
fix ui app name prompt description
feat add option merge to command create
fix make sure router js passes airbnb lint
fix tasks parameters
test bump jest timeout since vue eslint config are now externalized
test fix visibility issues in cypress test
ci bump test timeout
fixup test use page waitforfunction for hmr testing
feat support pug as html template
fix vue add should not throw if plugin has no generator
fix fix a regression that pages doesn t allow entry only string format
fix nightwatch fix globals js import in config
fix ui yarn link vue ui
fix runtimecompiler default
fix ui undefined sources error
chore tweak todo
fix fix pwa info link close
fix remove console log
fix eslint load node env by default fix
fix asset output relative check
feat support style lang postcss
fix cypress allow users to update cypress
fix ui task logs performance regression
test fix test errors
chore pre release sync
feat ui allow partial graphql results
chore fix typo s in comments
fix tests for windows
fix ui default tasks regex closes
feat vue cli choosing to save as a preset tells you where it is saved
fix hot hotonly client check
test fix test case on windows
workflow correctly set releasetype
feat default preset save to no
workflow fix test script p flag
fix serve task unset host and port default values closes
fix babel preset app avoid corejs warning when usebuiltins is false
feat ui basic global error handler
workflow allow release to local registry for testing purpose
improve package json not found error
fix cli service global fix no debugger rule config
feat support use h show detailed usage of command
chore validates multiple arguments for vue create command
feat warn when user modifies output path directly
test fix cliserviceversion for generator unit tests
fix avoid hash collisions
fix ui build task defaults should respect outputdir option from config file
chore add vue cli in version output to avoid confusion
fix babel preset allow setting usebuiltins to be false
feat cli service global to know if the build has finished
fix should not show error message when pnpm is not installed
fix better error message for missing key entry in pages config
refactor use console clear to clear the log
workflow document release process ci skip
fix webpack dashboard support any command for mode
chore tweak build watch mode message
fix mocha avoid inspect brk flag clobbering other values
feat lint default to lint tsx files
fix cli service drop webpack noemitonerrorsplugin usage
fix avoid scrolling when picking features
test bump pwaplugin test wait for ci
chore use stdout columns to determine the stats table width
chore pre release sync
chore pre release sync
chore pre release sync
test fix jsx test
ci bump invoke test timeout
fix ui bus plugin error
fix make extension test for font files case insensitive
fix when executing multiple actions only check git status once
fix eslint invalidate the cache when eslintignore changes
normalize file paths before postprocessing
chore minor typographical fix
workflow when syncing local plugin deps use minor version
fix ui create package manager select
test properly close server
disable prettier option until vue compat release
chore fix inspect
fix set url option for jsdom global
chore remove unnecessary log
chore pre release sync
test fix polyfill test case
fix vue add should not crash when executed under yarn pnp
feat use more descriptive classnames for css modules
fix better error message for non existent polyfill names
feat babel forward preset env s debug option
feat do not write undefined fields to config files
ci bump timeouts
fix pass all parameters to onproxyreq in proxy configuration
fix upgrade all avoid accidentally writing outdated package json back
fix audit hasprojectyarn
feat output help information on unknown cli commands
chore remove extraneous exclamation marks
fix ui args dedupe crashing if an arg isn t a string
fix fix stats display when chunkfilename contains query string
workflow force publish all packages
fix pubsub set max listeners to infinity
fix do not update dep if latest tag is older then specified
feat ui auto clean projects list
test remove only
fix correctly catch errors of offline install
chore pre release sync
test also test ts babel jest
feat check for nightwatch conf js config
fix cypress stop ignoring config file cypress option
fix force babel core version when using ts babel
lint on save
test fix jest args order
fix avoid emoji length problem in update log close
fix temporarily avoid chalk color in options
chore pre release sync
fix ui progress handler should not throw error casuing process to exit
fix ui prompt confirm checked
chore fix typo ci skip
fix babel legacy decorator
docs rephrase the upgrade notice
chore skip cd command in finish instructions if created in current dir
fix fix a typo that caused router failed to install in older versions
fix dev server dev server behind nat network
fix ui dashboardplugin ack data
fix should not thrown on windows when proxy target is an object
fix should check yarn when no pacakage manager specified
fix assetsdir can be an empty string
chore fix typo ci skip
feat cli allow local json files for presets
test fix assertversion test for prereleases
fix ts prompt spec
fix fix typo
fix only provide baseurl fix if baseurl provided
fix default html template closes
chore show network unavailable for explicit localhost
chore deprecate tslint
fix cli plugin unit jest also process svg files with jest transform stub
feat bump default less loader version
chore cleanup vue config ci skip
fix force lines to be ended with lf fix windows tests
workflow correctly decide which dist tag to use
fix fix a bug that caused mode to be ignored if run together with watch
chore pre release sync
fix names of sass and less
test fix ts babel test loader assertion
test bump timeout fixes appveyor ci
test fix publicpath import typo
test bump timeout
fix remove the nonexistent test command from generated readme
fix correctly shim vue compiler sfc for fork ts checker plugin
chore disable gql linting
fix tslint also lint tsx blocks
fix test name
fix ui pwa config manifest json indentation
chore prefer tsx files over jsx file extensions
fix typo in plugin invoke notification
chore update cypress test js test description
test asynctogenerator is no longer used in async await transformation
fix eslint generator add ts file check to lint staged
fix fix support for some legacy registry servers
fix revert default symlinks setting
fix force enable jsx in parseroptions fixes
chore add warning when using modern with non app build
workflow should use follow tags when pushing
fix fix cssnanooptions format
fix ui force output colors
fix hash module ids in anonymous chunks avoid enametoolong error
chore minor typographical fix
fix ui client addon serve
test don t check for taobao registry during tests
fix serve also detect kubernetes containers
feat ui enable vue devtools
chore pre release sync
test force click finish install as there seems to be a bug in cypress
fix include test tests for eslint loader
chore pre release sync
workflow fix the maxlistenersexceededwarning
chore pre release sync
feat set pretendtobevisual option for jsdom global
fix eslint autofix code style after scaffolding on older versions of cli
fix use yarn version to detect yarn
fix eslint save on commit test
fix generator add core js as direct dependency of generated projects
fix ui shareddata watchers
fix fix reference of friendly errors webpack plugin in config
fix ui remove console log
test skip version comparison test for now
test ui increase project create timeout
fix ensure loaders exist
fix ui chdir
fix rename test utils shallow to shallowmount
fix unit jest fix vue coverage report when babel plugin is not enabled
fix use devserver public to build sockjsurl if defined
fix ui ui command display url
test fix mock module in upgrader test
fix don t prepend publicpath with slash
ci bump timeout
test bump write wait on ci
fix typo in error message
handle copy plugin for global build
fix restore offline option for legacy vue init api
fix fix tslint on vue file with no script
chore improve vue inspect failure message
test attempt to fix multi line match for windows ci
revert chore pre release sync
fix local plugin remove folder before copying
fix ensure cache loader apply to both babel and ts
fix mayproxy ispublicfilerequest judgment
fix fix redundant log messages from webpack dev server
fix ui plugin actions
chore eslint use babel eslint
revert refactor use console clear to clear the log
fix temporarily disable space infix ops for eslint ts
fix don t output warning message on eslint deps upgrade
test bump test timeout for upgrade tests
docs document vue init clone option
docs enable algolia search
feat cli service add stdin flag to build
fix audit not implemented for pnpm error
fix cli service global fix eslint loader config close
feat should support bare option in preset json
workflow fix sync script
test ui disable locale watching for tests
fix update all plugins
fix should repsect configurewebpack hooks when resolving app config
chore eslint remove name from globals
fix fix hmr hostname when devserve host is set
feat cli add stdin flag to serve
fix ui configurations broken package json mode closes
chore ensure clearing env var in normal build
fix fix severity config in ui
fix module name case
feat xdg compliance rc file location hierarchy
fix ui add projecttypes to addview api
fix generatorapi remove warning when using extendpackage with prune
fix restore route making a view unnavigable
workflow allow version number as commit message
chore shorten current directory prompt
chore fix duplicate test project
fix fix kubernetes container detection
chore check file existence first in moveplugin
chore pre release sync
workflow fix release script
fix cli service fix polyfill injection when overriding entry with the cli option
test yet another dirty fix for verdaccio testing
feat add create option for router history mode
fix cli ui folder list when root windows
chore remove trailing comma ci skip
fix fix stylus config
fix ui process exit guard
refactor should resolve local json preset first
fix add missing package keys to the sort array issue
fix ipc check connection
chore also show typescript version in vue info
fix ui reset apollo store on reconnect
fix inspect plugins should log plugin names from webpack merge
feat ui log store reset
fix ui throttle dashboardplugin progress updates
fix create prioritize preset name present in vuerc
fix build wrong arg name closes
fix make cypress config file compatible with eslint config airbnb rules
workflow should be able to publish to next dist tag in local registry
docs fix vuepress link to next branch
fix apply a workaround for the prettier jsdom global bug
fix do not extract vue config js in tests
fix cli service treat specific flags as boolean only
fix unit jest issue with jest configuration regex not matching template
feat support custom package manager
chore fix typo in version check message
fix disable chunk sorting in html webpack plugin
test increase timeout as appveyor is being very slow recently
fix ui prompts missing await
fix check for project pnpm only
fix should not inject babel core shim if babel plugin is not used
fix use sync fs methods in writefiletree
polish word mistake
test bump pwaplugin test wait time
fix ui remove console logs
fix should use the local version number if the cache falls behind
use normal link on ci
fix do not swallow vue config js errors
test fix baseurl normalization
chore ui clean up console log
fix add default inmemorycacheoptions
chore cli bare flag should take higher priority than preset
test uncomment the airbnb eslint migrator test
fix fix force flag
make hmr work
chore fix lint command description
fix fix local preset inference on windows
test ui fix linter enabled by default
feat respect devserver field in webpack config as well
fix ui unhandled auto project open error
ci disable thread loader on circleci
test fix stringifyjs import
chore remove outdated warning
fix ui client addon config
workflow try defaultmaxlisteners
fix plugin api task hooks non vue cli project
test fix pwa spec after defaults change
fix fix pwa ts lint close
fix ui client addons serve
fix fix resolve project local plugin s file path
fix serve respect devserver openpage field
feat enable caching for uglifyjs plugin
feat bail when user directly mutate output publicpath
test do not call process exit during tests
fix properly load non js config files in gencacheconfig
chore add newline in prompt message
test fix objectspread test
fix fix missing s in directory path
fix tslint don t change working directory
ci bump test timeouts
feat make globby includes dot files
test do not assert publicpath mutation during tests
refactor fix tests service run should return promise
fix fix false positive of hasprojectnpm
fix eslint config versions
workflow don t skip version prompt for local release ci skip
fix fix usage of cmd shim
refactor remove unused placeatrootifrelative parameter
chore dev server set clientloglevel option to silent
ci auto detect chrome version
test remove unnecessary log statement
fix fix options for css optimize plugin close
fix ui eslint lint on save default
feat ui babel feature description
docs add description of babel plugin usage in ts plugin prompt
chore pre release sync
fix add absoluteruntime option to babel plugin transform runtime
fix should pass cwd as argument
fix add timeout for requests
fix create commands add by plugin through preset won t have description in readme
fix tasks unset options
fix build target lib should merge externals in vue config js
ci avoid jest bailing on appveyor
fix include prereleases for assert version calls
fix ui remove eslint disable comment
fix use module directory instead of entry file for eslintpath
chore remove log ci skip
fix missing proxy argument
ci bump timeout
fix correctly calculate cacheidentifier from lockfiles
fix ui watch only locales folder if exists
fix ui reset webpack config js service on correct cwd closes
fix should not proxy sockjs endpoint
fix wc don t overwrite user defined externals
fix fix version check
fix remove debugger
fix ui remove ui from built in plugin list
feat add decoratorsbeforeexport option
fix fix overwrite prompt
fix take all env variables into account in gencacheconfig
chore ui clean up console log
fix ui open project check if folder exists
fix upgrade correctly update version range in package json
workflow tag generation for vue cli version marker
fix ui task arg deduplication with
docs add browse plugins link to header
fix fix test issues for cypress gui
fix do not install core plugins that have major version bumps
feat support webp
run puppeteer without sandbox on ci
test fix eslint tests
test should not use file protocol now that we have verdaccio
ci revert unintended change
fix only enable helper aliasing when usebuiltins equals usage
fix graphql type json has changed its commonjs export interface
ci bump cypress timeout for appveyor
fix ui clear client addons on pluginapi reset
test fix helper esmodules for jest
fix audit handle errors
fix ui process killed before ipc messages are sent
fix cli test utils fix initgit option
fix ui webpack dashboard fixes
fix avoid injecting polyfills when targeting node
chore cli plugin typescript fix typo
fix build only modify css extraction if its enabled
fix ui cli service more info link
test allow generatewithplugin createtestproject to be invoked in the same file
fix ui unset last opened project if it is remove
test ui increase timeout
fix fix friendly error plugin name typo
feat enforce emoji variation in logs for gear
fix eslint should override env for tests subfolders
fix ui don t split on
fix ui use separate db for dev
fix fix eslint path resolution in vue serve
fix plugin generator should be optional
fix fix eslint loader for typescript
fix import createtimeago from vue timeago because it isn t a default export
feat don t throw on console debugger statements for vue serve
fix cli service avoid opening browser twice
docs fix typo in error message
fix respect dest when copying static assets close
fix lib replace url module with simple string split close
chore todo ci skip
feat support node nightly builds
workflow do not log if link check is successful
feat add ability to use environment variables in vue config js
fix fix ts tsx rule separation
fix ignore svn when reading and writing files
workflow re enable pre release sync
ci log patched chromedriver version
fix eslint plugin module loading in workspaces
fix allow console during dev
fix avoid crashing when add router via ui
fix fix duplicate slash in metadata url
fix css sourcemap in production
chore pre release sync
fix proxy should warn when array is passed
chore no longer needs to disable space infix ops
fix service require path
fix typescript fix typescript multi page build
changing tabs to tab
fix task remove non running missing tasks
chore ui comments
chore do not clear screen when serving so that warnings can be seen
chore always upgrade vue cli service first
fix ui serve disconnect ipc after first run
feat better validation error message
feat enable hmr for mini css extract plugin in dev
feat allow configuring postcss loader via css loaderoptions postcss
feat add mjs files as transpileable files
ci bump timeouts
fix fix dev mode compatibility with workers
fix typo in vue loader compileroptions
annotate config js
fix compiler fix codegen for v for component inside template
fix keep alive higher priority for exclude than include
test fix global filter registration
safer isnative check fix
vdom teardown stale directives on patch fix
chore remove unnecessary log in tests
avoid error when asserting mismatched nodes during hydration
refactor use more accurate variable name
make flow pass
fix test template typo
types fix flow typing
port hyphenate fix
fix allow codebase to be inlined directly in html
fix work around ie edge bug when accessing document activeelement from iframe
avoid double decoding on attribute value
improve proxy property warning
more informative warning message for comp name
nextrefactor test is faster than match for regex bools
fix further adjust max stack size
fix core remove trailing comma in function signature
handle root props
log render error instead of throwing
add assertion for enter hook timing
fix a typo
fix codegen support named function expression in v on
remove parent ref elm references when destroying instance fix memory leak fix vue router
add extra change listener for v model fix
fix identifier re
tweak transition appear check
chore fix comment grammar mistake
fix test error in phantomjs
make functional component children a thunk
work around flow
fix transition fix out in transition getting stuck with v if
fix select multiple warning test case in ie
support key modifiers
fix errorhandler tests
annotations for createfunctionalcomponent
render method should use dynamic binding
chore filter spec fix variable test case
fix missing component tag
use isserver flag instead of runtime detection
types make flow happy
fix unnecessary conditional
fix compiler set end location for incomplete elements
build fix chrome headless on ci
improve add missing html and svg tag
fix avoid breaking avoriaz edge case
tweak select ie fix to reduce duplicate call
reset slots on each render
warn against incorrect propsdata usage
fix update props
test fix assertion
fix typo in util debug
chore merge imports
fix parser only ignore the first newline in pre
fix fix server env detection in wechat mini program
refactor fix types
chore fix typo in next tick js comment
tweak v model type file warning phrasing
fix ssr not push non async css files into map
fix allow multiline expression in v for
chore fix tag name in compiler parser index js
fix test case in ie
test remove unecessary id in test case
fix attr codegen
fix vdom remove unnecessary samevnode condition
fix flow error
refactor remove redundant ternary
improve bundle renderer file path detection
fix model fix array index binding for v model checkbox
remove unnecessary alias
coverage for nested svg vnode namespace
remove no longer necessary code
fix ssr should not optimize root if conditions
fix keycodes proxy set return value
test weex avoid v else warning for recycle list tests
move namespacemap into web util
fix events properly off array of events
allow script tags
fix linting issues
fix fix ssr env detection in weex
fix ssr expose context styles when no lifecycle styles are injected
fix prop validation incorrectly setting observerstate shouldconvert
test add a unit test for off event and this event add by once
test improve inject default test case
adjust keep alive name matching priority
fix ensure proxied normal slot uses correct key
fix weex tests
simplify source with rest params
refactor tweak dependarray change
fix eslint and coverage
refactor simplify html parser attribute regex
fix make transition group key warning a tip to avoid breaking compilation
refactor improve test case
chore ignore code that cannot be reached in phantomjs
chore fix flow
update util js
improve mergedata for edge cases fix
fix multiple merged vnode hooks not invoked properly
fix v model handle trailing whitespaces in expression
chore platforms web server util removes duplicate values from the attr list
update track by usage
weex use weex requiremodule instead of weex require
revert refactor remove unnecessary checks
avoid swallowing webpack error when entry is not found fix
test improve reserved props test
fix avoid unnecessary lowercase coersion in component name validation
fix keep alive nested child components
fix ssr add semicolon before self removal script
settimeout fallback update
ensure scheduler depletes all queues
fix proxy has return value
chore fix typo
use genassignmentcode in chehckbox model web compiler
expose oldvalue on directive bindings
small tweak to filter parser
fix uglify upgrade
set default value for ignoredelements to
only check keep alive on dynamic components
fix hydration node match assertion for elements with camelcase names fix
chore fix types
better template render warning
fix transition appear check for nested components
fix compiler handle negative length in codeframe repeat
fix use mixin api feat continuous operation
remove bind prop shorthand
fix linting problems
fix block unnecessary input event on input tag placeholder in ie
no need to check domprops in extract props because domprop casting can only happen for native elements
chore fix flow
filter out empty strings when normalizing children fix ssr match when containing text is empty
fix prevent memory leak due to circular reference in vnodes
fix for use ie compatible regex in v for regex
handle include exclude changing to non valid types fix
fix weex tests
fix multiple events
expose component v model expression on passed data
throw error in ssr so it can be captured properly by renderer
fix ie edge select remembering selection
do not add list key to text nodes fix
remove name option warning during instantiation
chore fix typo in bind dynamic keys js comment
fix skip v model value binding collision check with dynamic type binding
fix functional test
weex fix the removechild logic for text node
fix allow an object s symbols to be observed
refactor use isregexp for safer check
fix ssr resolve server directives the same as on client
wip fix codegen
fix incorrect unknown element warning fix
fix props correctly warn when a provided prop is symbol
test for errorhandler config
fix provide provide should default to parentval during merging
refactor ignore control regex lint for unsafeattrcharre
fix component v model with value binding expression fix
fix data skip recursive call if values are identical
warn computed properties that are already defined as data or props
fix include boolean in isprimitive check
fix edge test case
locate first non abstract parent as parent
fix weex test case
compiler codegen improvement
avoid generating empty ssrnode
work around ie textarea placeholder innerhtml outerhtml bug fix
fix tests for global name override
fix ensure only nromalize a scoped slot when it is present
fix child component destroy
test change model text s priority case
use samevnode check when pacthing keyed children fix
fix delimiters caching fix
bind value as property for select fix
render should be a reserved tag too
fix handle arrays in v on object syntax
refactor keep single source of truth
v show should ignore inline display none ref
fix ssr fix bundlerenderer mapped async chunks caching check
ignore ssr getter
fix check entertoclass leavetoclass existence before add it
refactor should also fire handler if timestamp was same
fix handle special case for allowfullscreen on embed
test add test for ignoredelements
fine tuning of tonumber function
fix delcare typo error
fix test in phantomjs
add support attribute module
refactor argument name
avoid updating input value during ime composition fix
fix ssr test case for init beforecreate change
chore fix flow
fix prevent default unit test
fix hydration warning test in phantomjs
comments for flow type nodes
store v show original display value directly on el
polish warn against incorrect nested v slot usage
fix ssr fix double escaping of ssrnode attribute values
chore fix typo in element test
avoid ssr build warning when using minified build
typo fix optimizier optimizer
chore fix flow
fix scheduler getnow detection can randomly fail
refactor do not override scoped slots when exposing normal slots
remove font from issvg detection to avoid issues with legacy markup using the tag in html close
remove transition text node warn spec
add the missing break
fix v show without transition
remove extra paren in warning
fix text parser regex for multiline tags fix
fix functional components that return string or nothing fix
fix transition group work around rollup tree shaking
fix filter parsing for divisions fix
fix ssr avoid missing files in manifest
prohibit mounting to body html
only call query when in browser
simplify v model assignment gen support add reactive field when using dynamic segments
chore cover more cases in ssr bench
replace special terminators chars
handle circular reference in looseequal fix
fix inject change warn message when trying to mutate an injected value
feat warn if both v model and v bind value used on same element
clone iscomment when clonevnode
line width tweak
fix properly render value on progress in ie edge
rename sfc parser export
include local tag in keep alive cache key fix
fix transition on appear
add csp warning
chore typo fix comment typo
fix the placeholder key in undefined when render async component
fix duplicate attribute test case in ie
check vm el presence first
improve vue config keycode typing
exposed compile function should be cached version
fix core only unset dom prop when not present
refactor set should return early if original def has getter but no setter
refactor avoid leading semi
fix transition mode on keyed normal elements
feat weex wrap iffe for appcode
feat expose performance measures
remove reference to refelm after initial patch fix
specify that tags with side effects will not be parsed
comment spelling errors
refactor remove unnecessary condition
chore minor comment cleanup
remove unused var
component vnode tag should be related to the constructor id
refactor remove unnecessary checks
refactor return is better when inject has no value
ssr cast node text to string before encoding
ssr fix json bundle path fix
test create component hook merging
test for v cloak
change default error behavior to throw
weex expose native module getter on sub vue instances
no need to import vnode type when its imported in declarations
update duplicate key warning message
weex expose version and config on subvue
adjust ignoredelements check after merge
revert remove no longer necessary code
limit v model runtime shim on text inputs
build tweak release note file default name
fix make sure global state is restored in the case of an exception in macrotask callback
warn undefined render template during ssr
refactor add empty v bind warning re
add the runtime only warning for el in dom html template
fix addstylebinding helper
test add a test case for ssr max stack size limit
do not use looseequal for checkbox v model if has no true value binding fix
add unbind hook for runtime directives
fix observer do not invoke getters on initial observation
fix flow missing annotations
first component rendered
fix ssr fix v show inline style rendering when style binding is array
fix options for inline template instances
fix replace hardcoded parentnode with abstract ops fix
ssr relay renderstream error in bundle renderer
improve attr interpolation warning
ensure transition end detection only count events triggered on target element fix
fix watch option in firefox
adjust constructor option resolution timing
fix ssr address possible xss vector
forbid tags with side effects in templates
add picture to ishtmltag map
improve dom listener update performance
chore typo in variable name
update index js
chore fix flow
feat feat add warning for ambiguous combined usage of slot scope and v for
fix ssr bundle renderer error on commonjs exports fix
chore fix grammar in core observer index js comments
fix template slot
remove html encode cache in ssr
allow rendering forbidden tags on the server
fix addclass utlil bug
test make hydration spec more stable for edge
destroyed elements pass their styles onto others that are similar
ignore untestable check
process transition in compiler
chore fix comment typo
fix v on unit test
chore fix warning space
fix runtime devtools recommendation shows for all browsers
skip v show transition if value is the same fix
tweak type definition layouting
warn empty or missing template element fix
fix model fix static input type being overwritten by v bind object
test vdom add test case for
fix flow def
allow registering custom compile time directives
perf deep clone slot vnodes on re render
avoid traversing frozen objects in deep watch
test add test for observer lazy getter invocation
fix sfc component contains only
remove v bind style important test case due to phantomjs inconsistency
remove no longer needed built in tags
only handle script preload
fix v on handler with whitespaces fix
fix skip microtask fix if event is fired from different document
test use headless chrome for dev
refactor use istrue
adjust saucelabs browser list
feat weex return instance in createinstance
ensure custom directive bind is called after node is inserted close
basic mathml support
feat ssr auto remove initial state script if prod
complete coverage by testing v model with extra listeners
fix doc comment typo
normalizechildren should pass down namespace recursively
refactor empty function replace to noop
chore s match all whitespaces
fix name warning when instantiating extended components fix
avoid pushing the same ref more than once fix
fix properly mark slot rendered flag in production mode
fix node ops type declaration
avoid calling duplicate enter leave when already in transition fix
refactor remove ternary operator because key is always true
fix hyperscript direct string as children
fix eslint no self compare
fix out in mode duplicate calls
fix typo in comment
fix ssr tests
add foreignobject as a dedicated svg element fixes
chore fix lint
refactor reduce component mergehook arg count
refactor core remove unnecessary switch case
include component file location in warning if available
fix eslint in tests
remove filter from config asset list
allow require as a global in proxy check
improve proxy detection
fix enumerated boolean attr
give long transition test specific timeout value
support using extended contructors as mixins
chore typo no dots at the end of the comments
types add fragment in renderstate typing
refactor compiler remove head tag contains html tag condition code
cache vnodes length
ensure movecb only gets called on transform complete
fix platform compiler directives
small improvements on html parser
fix ssr reference error when create ssrcontext for root component
refactor codegen only generate pre data for possible components
test computed property conflict warnings
remove no longer relevant code
avoid showing chrome devtools message in edge
fix weex send createfinish signal after root component mounted
better out in
weex set append tree by default for cell component
fix test for emptyvnode rename
prevent data pendinginsert to keep reference to remove nodes fix
chore avoid production tip during tests
remove transition text node warn because the wrapped component may return empty nodes
fix v move class when name isn t specified
chore trim trailing whitespace
reference doc in component data function warning
no need to return text in getvalue for v model select fix
suppress a warning unknown custom element with ssr img tag
fix ssr v bind test
chore model text spec add a missing check
fix typos in src server bundle renderer create bundle runner js
chore fix typo
test for static style important support
optimize whitespace vnode
use console error as default behavior
fix set value as domprop for progress
trim ending whitespace
weex fix the text node could not be remove properly
give the component update computation a name
revert fix select multiple first option auto selected in chrome ff fix
fix v on patching for cloned slot nodes fix
avoid html decoding empty strings
test directive binding expression
bump weex test deps
small tweak for coverage
fix hydration tests
fix work around old chrome bug
handle enumerated attrs
refactor also clone vnode asyncfactory
fix ssr hoc
always process key
fix style important spec
improve readability of updatechildren
only mount child component to existing element if hydrating
fix runtime directive unbind
handle important in static styles fix
expose warn as noop in production builds
test undefined method warning
refactor also use emptyobject for functional listeners
fix transition transition group should only listen for first level children s end events
tiny comment fix
global mixin should not drop scope id fix
update version mismatch message
test observed vnode data warning
update track by usage in test
fix compiler fix inline template crashing
polish improve template compiler version mismatch warning
stricter symbol support check
make v model checkbox replace array instead of mutating it
fix duplicate slot warning spelling
add computed property ssr test case
fix always set transformed model value on attrs
fix support modifier combination of click right once
call pending entercb before move
chore preservewhitespace is true by default
refactor remove unnecessary judgment
chore typo in comment
fix directive rawname generation for raw render functions fix
fix v show transition on appear
refactor improve error msg for non reactive properties
should not generate children for is component with inline template fix
overwrite static node key in transition to ensure unique pending key
fix native events on components with inline template fix
include templates in ref spec
refactor mark weex specific block
rename init beforecreate
avoid using native as identifier close
ignore deprecated alias
fix codegen import
fix weex support data class type that is string
update util js
fix ready hook
remove unused before hook
fix fix function expression regex
fix parser allow crlfs in string interpolations
fix avoid ispromise check when handler return value is vue instance
fix avoid logging same error twice when thrown by user in global handler
remove forgotten debugger statement
should only call updated after mount
make foralias regex lazy fixes
fix resolve async component default for native dynamic import
fix using one delay to control all transitions
fix html attribute escape in safari
increase async delay for more test cases
fix off should ignore undefined handler argument
fix keep alive non component child
check mutationobserver native ness
build unexpected token when running dev
chore improve license clarity of simplehtmlparser js
fix global component name
toobject should not mutate original objects fix
perf core prevent iteration of arrays that should not be observable
ensure enter and leave transitions happen simultaneously
feat support v html for svg elements
add vm annotation in init
use umd for cjs build size check
fix allow traversing reactive objects which are sealed
no need for unbind in model runtime
fix class initial application
fix flow error
revert support transition on component with v show in root node fix
fix sfc parser hanging on trailing ill formatted brackets fix vue loader
fix fallback to promise in non dom environments
fix runtime build exports
only preserve whitespace if it is not right after starting tag
chore fix setup script
update outdated message in ssr plugin
test fix test case for
watcher errors should be handled in production mode as well
reuse no function from utils
coverage for sfc parser
fix do not traverse vnodes when regsitering dependencies
refactor minor tweak on conditions order
test template key warning fix coverage
fix compiler maybecomponent should return true when is attribute exists
build update release note script
fix comment typo in normalize children js
refactor add missing else in gennode
ensure prop default value when instantiating without propsdata fix
support direct string as children
include pre post transform declarations
tweak comment of coverage setting
fix transition appear check again
fix global mixin recursive name lookup
chore fix some closetag mismatches
refactor codegen simplify conditions
support arrays in vue set
fix static node v for nested check
fix transition handle local formatted floats in toms function
test for vue compile
fix ie conditional comments fix
fix render codegen bug
actually complete test case for
fix ssr fix bundle renderer require path on windows
refactor tweak data merge strategy
refactor use util isnative to check native proxy
revert keep alive should apply in all cases
observer conversion enumeration should not include prototype properties
fix value prop handling
remove unnecesary function
chore fix flow
perf cache result from functional ctx slots calls
fix properly without hacking the type system
chore fix prefixed data properties warning typo
test add test case for
fix spec fit
fix select multiple first option auto selected in chrome ff fix
more robust parent transition skip check
use same alias file
no need for escapenewline
better infinite loop warning
test fix lru upgrade
fix hot reload in ie in strict mode
fix default slot fallback on whitespace
feat support regexp in ignoredelements
test for ssr rendering async components
fix improve error detector v for identifier check
chore check binding value instead of just binding object
test use phantom for dev
fix codegen support filters with in older browsers
fix warn slot scope when used as a prop
test remove unused code
fix show prevent transitions from starting on change truthy values
compiler should output code move function generation into mount
fix attr merging
fix address potential regex backtrack
remove preservewhitespace usage in dbmon
refactor extend dom props update skip to more all keys except value
coverage for new global mixin strategy
warn computed properties that clash with existing properties close
return emptyvnode if render function is not provided
improve add local variable declaration
fix xhtml incompatibility fix
fix ssr fix double escaping of staticclass values
fix weex model test
perf optimize the performance of hyphenate method
tune scheduler to ensure update order for late pushed watchers
createelement won t return array vnode
fix transition spec typo
warn observed vnode data handle in place mutations for vdom modules ref
fix transition on child component root node
inline export for consistency
v model modifiers
fix duplicate import
fix sfc parser unary tags handling
polish compiler remove the extra space of v for alias
fix watch option string
ci bump transition unit test duration
update test case
include exclude for keep alive
weex expose native module getter on sub vue instances
use extend instead of object assign since the latter is not avaialble in phantomjs
fix ssr for child components
guard input value as string for v model
add pre directive tests
check for drawn digits before sending data to the server
adjust ocr js file comment
remove extra line
add a range helper as per review from paulsputer
add constants for hex colors
fix repeat block to take blocks
some more comments
fix for serialization
add quotes around json keys
update url js
fix syntax error
ignore partial helper error
database add database query
database fix collection find
refactor cheerio is no longer necessary
correct the spelling from generatd to generated
remove target argument in rss migrator plugin
refactor set default pattern
docs codeblock fix line order
don t push if nothing changed
docs codeblock update mark description
output util to global variable
fix scope of the getid function definition for toc helper
fix generate watch bug fix
replace path sep with
generate only save assets collection
fix a typo in the hello world example post
fix grammar error in output of hexo init
remove callback parameter in paginator
load plugins when init
display generating time in three decimals
bug fix in page js
remove language prefix of code blocks
display more detailed version information
fixes typo in renderer swig was locales instead of locals
prints swig render error with file name
fix creating a new page
fix plugin loader
fix test error schematypemoment validate
call parse arguments with optimist
merge pull request from sukkaw drop lodash for date helper
database set default getter setter of virtual
docs codeblock document each parameter
refactor destructure util module
use instead of
refactor drop lodash merge
fix loading deployer of heroku and openshift
merge pull request from changyuheng pr localhost by default
fix tag extension bug
correct the ip address displayed by server
modified yaml front matter parser
add test case for the hour and minute of the post permalink
database fix date processing bug
omit s r properties in fromt matter
don t execute non init console when the directory is not init
test post fix cases add in
add a data more variable
async variable not set causing error
fix hexo route undefined
fix rebuild bug of home generator
comment enabled by default
delete property of rsync was not effective
fix model sort bug
display uncaught exceptions with log error
test generate add a small delay before clean up
use link as title if post doesn t have one
generators updated home
set scrolling no to make iframe responsive
changed filename escaping rule
add postid support
fix open graph helper
transform single tag into a string
set hexo cache preview in preview mode
not to add init when initializing
require tag cloud
write static file with fs writefile
merge pull request from jkuetemeier tag typo
don t add layout property when creating new article
change buffer api
change the built in server ip config to localhost
extend renderer remove prefix in output parameter
update tag js
yfm render yaml via render api
new type of extension filter
chore make linter happy
merge pull request from curbengh default lang
refactor test console migrate async await destructure
ignore plugins whose name is started with types
fix processor render error
fix tag code parse wrap option
fix tag cloud sometimes didn t order by options orderby
preview decode url
fix benchmark wrong measurement on hot processing
add test for swig template compile
merge pull request from dca master
fix test post publish
make swig capable for warehouse query instance
fix maxopenfile variable
break if not configured
rename view folder
merge pull request from curbengh helper class
refactor chai useful should own property
try catch for sync renderer
case sensitive on github page deployment
remove array check on variable that is always array
undefined variable in deployer util
changed language code
fix server s static
server enable error handler in debug mode
feat toc seek anchor id from parent nodes when target is not available
logger detect error with property
fix server cache must match exact alias
fix preview console mime
add draft option to hexo server help
fix to more simple way
merge pull request from sukkaw drop loadsh for toc helper
fix render console
should not clear database when use hexo new hexo help etc commands
fix migrate cli
add global variable hexo env args for cli arguments
accepts string as plugins setting
modified inheritance way of model
merge pull request from kenspirit master
new way to define category set categories property in posts
coffeescript highlight alias
style router clarify function name
add page locals for category tag archive
add whitespace after tagcloud links
add cache in hexo namespace
fix script loader
correction of wrong auto substitution
add parameter to disable highlight auto detect
fix jscs error
style remove newline
add end tag of js helper
test post add extra test cases
refactor renderer arrow function
ignore empty category tag
test tags code update mark test
refactor migrate console
fix new assets are flag as not modified fix
merge pull request from heroicyang dev
transform backslash in route into slash to fix preview bug on windows
display logger in debug mode
throw error on missing partial
modified console info
merge pull request from oncletom fix rsync delete
chore fix lint
escape in backtick code block
post create fix event and callback can t be called at same time
merge pull request from heroicyang master
display info when watching fies
display os information in version console
modified spawn until
fix tag cloud helper
use bluebird tap
preview lower priority of public directory disable gzip compress
exit process manually after generation done
redirect to admin if users has logged in
execute callback function after hexo list route json done
merge pull request from curbengh og locale backward
add comment on why we re truncating update test to use should
display theme source file render error directly prevent error callback was already called error
merge pull request from hulufei master
merge pull request from oncletom fix more filter
database add query tojson query stringify
trim instead of regular expression
refactor utilize encodeurl of hexo util
display information when theme assets installed
should listen to localhost only by default
display version information when using hexo v or hexo version command
perf avoid running plugins in clean command
server allow head request
remove meaningless line
fix permalink parameter in article not work
merge pull request from juanitofatas fix a typo
fix size filter
model assets source assets path
remove comments in util test
creating index html instead of placeholder when setting up deployment
fix process uncaughtexception log
yaml front matter ignore the first blank lines
fix error referenceerror moment is not defined
getinfofromfilename maybe can t find year month date from filename
merge pull request from sonald fix typo
restricted trimming of tag bodies to the presumably intended cleaning up of occasionally parser inserted extraneous line breaks
database copy property not in schema
apply console options
unescape in backtick code block
add after pagination link
remove path parameter
fix util escape filename which used to change filename to lowercase forcibly
fix copying useless data when generating
fix ci error
transform generate console argument into lower case
modified callback arguments of route update
merge pull request from segayuu change buffer api
fix files in source folder not copied to public folder
create createlink function
remove unused module
post create wrap title with quotations
fix util spawn
add engines parameter when init
use destructuring assignment
fix server cache must match exact alias
add semicolon back
expose render function
fix schema function undefined when init
rewrite yaml front matter
fix external link filter
modified clean console
add test routes
merge pull request from chenall dev
reset theme configs when deleted
util timer util timer
modified file api
fix theme getview
theme config inherits root config
fix lint error
processor fix escape function
database modified collection get
delete useless parameter
fix layout of command list
load database from db json
fix search to use sitesearch param instead of two q params
add hotkey esc for exiting fullscreen
use promise each in box process to prevent race condition
display generating time in three decimals
fixes meta generator detection
fix cache key name error for box file test cases
style tag return callback if called
fix box always ignore git folder of themes
rename hexo in package json to hexo site
fix default disable highlight hljs by default
modified debug log contents
fix box pattern
modified init configure
update tag js
edited info of github deployment extension
domain module fallback
update tag js
modified content of hello world post
apply filter extension to ejs swig
more detailed error info
refactor test hexo destructure
remove useless code
merge pull request from curbengh tag code wrap
fix typo in console server
merge pull request from sukkaw fix test article tag
configs are now expanded for hexo util highlight to make it easier to expand upon plugins
merge pull request from changyuheng pr correctness bind ip
don t load plugins in safe mode
new option level
refactor drop lodash for new post path filter
fix tag cloud
fix moment language
the rauthor variable has been deleted
modified os info in version console
merge pull request from inetfuture fix
fix wrong arguments in deployer util
more strictly check for category duplicaton and tag duplication
unit test for highlight disable option
fix wrong category permalink
fix yaml front matter
updated markdown renderer
fix theme assets copy
modify layout order of page post generator
fix yaml front matter
fix batch deploy fix
package json indention
processor fix miss some files
check if deploy exists
util highlight add csharp alias for c
check if public folder exists before loading cache
add env variable to global variable
merge pull request from wolfgangkluge relative url
add swig extra to the swig renderer
disable login in debug mode
fix render read source file when data text is a blank string
route destroy route remove
fix https warning
modified console info
protocole aware vimeo video url
fix codeblock closing code fence may be followed only by spaces
refactor theme view drop lodash assignin
fix typo remove remove
add prev and next property to every post
fix category slug
fix files can t be processed after error occurred
use js yaml to parse yaml front matter
create scaffolds folder when init
refactor early return
refactor gravatar utilize hexo util module
style remove unnecessary argument initialize
add lodash to theme variables
fix deploy copy bug
cli remove v from help
watch don t filter type
fix the bug of css
fix rener js
modified default values of paginator helper
fix draft rendering
fix a bug in which categories were not paginated
fix post render can t process swig block with arguments
merge pull request from sukkaw meta generator
fix incorrect data type
fix tag plugin parsing error
modify yfm regex
sort command list alphabetically
add options argument to spawn
fix bug when add non article file in source folder
modified the content that route console display sort routes alphabetically
fix article rendering bug
add size filter
analyze repository url
fix process didn t exit in debug mode
replace new lines by spaces in description
add cache property in hexo namespace
record remote branch in config
fix asset id change source config the source assets folder can t be generated
remove comma first var declaration
makes swig rendering synced
fix error occurred when glossary not found
tab to space
processor escape in regex
fix allow backtick code block in blockquote tag plugin
fix deploy console
append package json when initializing
docs extend helper bring up jsdoc
not to display error code when error occurred
update scaffold js to use the latest style of yaml
merge pull request from aulphar master
fix og locale ignore case flag is not necessary
only render error page when request accepts html
merge pull request from sukkaw meta generator config
modified route format rule
rename raw tag to rawblock
transform some data into buffer
fix init console
modified info of init
use eventemitter to rewrite route
fix articles disappear after file changed
add more info to console help
deleted useless code
a better way to delete continuous line breaks
docs codeblock wrap option
test add test for number of lines in code block
merge pull request from curbengh refactor clonedeep
fix a typo
fix error multiconfigpath is not a constructor
database transform string to number with prefixed
processor write source file with stream
update lib plugins helper tagcloud js
close layout cache temporarily
perf titlecase lazy require
fix infinite loop
allow theme config to be overridden by site config
changed not to use array slice
fix tag path
strip the last slash of config url and append a slash to config root
merge pull request from aexmachina toc with no headings
remove useless code from preview console
fix model schema bug
throw error in server
fix issue add tail to archive links
merge pull request from curbengh db log
use date instead of file mtime for updated date
fix render helper
file add empty function
tag fix special tag names
merge pull request from curbengh console clean
add site to hexo namespace
fallback for git failed when init
fix deploy console
merge pull request from oleggromov swig typo fix
fix remove duplicated default path
database get data from store when updating
fix error occurred when scaffold folder doesn t exist
test box fix test cases for macos
fix set english as default
skip some tests to fix the timeout problem
fix for relative url
add path and permalink properties to paginator class
modified post index controller
display error message in red
category tag generator skip empty object
merge pull request from liuyl liuyl
findindex to indexof
refactor test helpers partial async await
merge pull request from oleggromov feature swig sync
add alias to random shuffle
read package json with require
delete hexo cache preview
don t load database when init
util highlight wrap each line with div class line fix
file watching modified filter regex
refactor test extend tag destructure
modified regexp escape rule of processor
apply route to load processor
delete useless code
disable auto detect for highlight by default
add slash after permalink of category and tag
remove unnecessary information in init configuration file
fix titlecase filter
remove json option of config console
util empty process separation in exception
database fix virtual setter
add category tag name in archive
root cannot be empty or undeclared
fix server error when file is blank
style eslint arrow parens with as needed
fix ignore theme installation
fix test error
fix mistake in tagcloud js
fix wrong category of post
add raw tag test
add option for first line number to test
fix error log
refactor remove site config from theme config
display detailed error info when plugins scripts load failed
fix config is not parsed sometimes
paginator class add base property
fix archive generator
expose theme configurations to stylus files
fix test time is omitted in front matter
merge pull request from chenall fix source dir bug
modified post schema and model
don t add permalink and path properties in every page temporarily
fix error occurred when tag is not defined
export std data with process
use markdown render content in blockquote
fix theme assets copy bug
fix tagcloud plugin result variable is not defined error
fix category slug
fix theme view processor
display render debug information
not to show error when loading plugin failed
use fs wrtiefile instead
add highlight to until
update test case
merge pull request from ileodo master
fix code style
fix date attribute default value
trim log message in log file
merge pull request from curbengh test console clean async
fix init option
do not compare undefined with null explicit the cas instead
execute hexo command with absolute path
merge pull request from tjwudi master
fix typos in plain js
merge pull request from hulufei master
refactor helper class syntax
add path parameter to article
fix ignore theme installation
refactor test console clean destructure
fix the bug occurred when category or tag is undefined
might as well add a configuration option to enable disable highlight all together
fix new event wasn t emitted
process theme assets after analyzing posts
fix theme init bug
server port fallback
remove tests server preview
refactor utilize hexo util module
modified help console
cache filename regexp
test remove empty test
create read stream after route called
modified server info
fix category link bug
merge pull request from curbengh clean plugin
test generate add a longer delay after fs unlink
changed plugin loading order
test re enable wrap
fix file empty mis delete git folder
merge pull request from chigix work toc fix
gist js file fix
files should be copied to deploy folder
test codeblock wrap option
fix callback not called in file dir when source directory not exist
load database when init
merge pull request from magicdawn fix special tags
fix paginator class
fix highlight can t be disabled
cast config theme to string
refactor arrow function
add rel next prev for links in paginator
load console plugin after init
remove tail in file name
fix toc use escaped text to generate toc from heading
delete extra blank line
changed filename escaping rule
route add clear method
fix database init error
add empty content test for yaml front matter
backtick code block don t process code block without any params
fix performance improvements
refactor test useful rest parametors
use spread syntax
remove useless code
image tag detect url started with
modified file watching ignore pattern
merge pull request from alxbl support hljs
fix render renderfile layout
merge pull request from magicianshiro master
post create add front matter escaping test
merge pull request from hcoona feature disable auto detect highlight
model add assets
fix generate console
update post js
patched loader with fallback
fix migrator fix help message
save log in debug mode
merge pull request from changyuheng pr built in listen ip
revert retain empty title behavior for posts
fix no post bug of archive generator
refactor cache option for cache rendered content
database fix and query
add something in page model
remove results attribute in search form
feat highlight parse caption option to prismhighlight
modified initial package json
generate export database after process done
diy directory is in current not
database simplify code
refactor theme sub classing
database add query update query replace
fix the issue
modified tag cloud helper
database add collection method
add json options for hexo list route
docs codeblock update description of parameter
remove useless attributes and add allowfullscreen
remove unneeded middleware from server
don t add categories property to posts if empty
emit exit event when process is about to exit
model fix pages excerpt
replace post page data instead update after source file updated
tuned generate render error message
fix excerpt bug
fix format helper
add options argument to tag plugins
merge pull request from jaredly master
fix markdown notation after backtick codeblock
database collection query add length getter
if config changed to mode
merge pull request from dailyrandomphoto fix save database
fix deploy bug
fix duplicated category
don t display help message in test mode
fix file empty
updating lib model schema js to match the more var
fix ignoring theme installation bug
test remove console log
close admin panel temporarily
refactor test generators page arrow function
fix extend renderersync register
fix generator fix site is not defined error
fix generating always ignore theme installation
database filter repeated items in an array
update debug helper test with rewire
before rendering yaml replaces all tabs into spaces
fix category generator typo
make admin always false
expose process in hexo namespace
fix markdown highlight wrapped by double pre code tag
merge pull request from curbengh hljs default
edit generate info
fix ejs renderer
merge pull request from sergiolepore add deploy events
update lib plugins helper tagcloud js
separate so that initial assign of option is done by object assign
merge pull request from segayuu use spread syntax
modify yfm pattern
alphabetical sort for collection sort
processor fix don t insert posts when creating a category
add source parameter to route callback
chore test update excerpt test case name
merge pull request from sukkaw perf regexp
generate fix old public files were not deleted
merge pull request from glittershark master
merge pull request from heroicyang master
fix space escape
fix server redirection bug
reduce path namespace object
syntax corrections for hint
protocole aware youtube video url
fix generate rebuild option
render pullquote content with markdown
merge pull request from mcfog master
only load plugins in initialized directory
append new content to cache
indent package json
apply highlight settings to code block tag
fix error when trying to initialize
throw original error object
merge pull request from jlhwung master
don t load plugins twice
fix multi line bug of backtick code block simplified code
change info url
fix logging log database only in relevant commands
fix the issue that code blocks in markdown are rendered as placeholder after disabling nunjucks
fix opens external anchor using rel noopener
database find in find nin transform data into an array
fix github deployment didn t add new file into commit
fix category path when placed on root
refactor drop lodash console deploy
merge pull request from chrisyip numeric title
highlight add more language alias
fix yaml front matter
fix toc helper would raise an exception if it was called with content that doesn t contain any headings
fix cache init doesn t callback when folder is not init
test fix new line
merge pull request from ferrari deploy gh
merge pull request from barograf master
fix rsync deployer
add hour and minute for the post permalink
fix test router format path must be a string
backtick code block fix remove non space indentations
test remove extra async keyword
highlight not backtick code block
generate call deploy console with call api
fix call api
feat helper hash url should have config url prefixed
fix lint fix lint error require atomic updates
add cache to gitignore when initializing
fix deploy console
modify deployer help text
change moment lang to moment locale
show elapsed time of each file when processing
support include css in stylus
return locals after analyzed
escape contents in backtick code block
remove class before source in blockquote
add deploybefore and deployafter events closes
remove unnecessary regexp test
remove filter indent buggy
style assign to this variable
add hexo cache destroy
fix copy bug
remove console log s to be consistent with other deploy scripts
wrap video with div video container
add logger option for server
delete unnecessary captures
quote attributes correctly for older browsers
merge pull request from neerajdotname master
short circuit on attributes
use more descriptive test names
remove namespace test from custom events test
add test for
use some instead of any
smart braddunbar simplification
fix comma at end of collections test
ensure options is present
merge pull request from braddunbar bind routes
processdata should use ajax default true and only be set to false when doing a non get request
reverting previous change
undelegate returns view
remove complex router text unfixable
grab id using get in url
merge pull request from trismcc fix model trigger
merge pull request from tgriesser isequalchange
merge pull request from wyuenho view native hooks
minor performance refactor
remove accidental global
just slice the root
fix opening and closing iframe doc for jsdom
add additional router test
only increment length once
add a comment about clearinterval
allow model matches to take a predicate function
don t mutate attrs
passing a configured version of jshint
call backbone model defaults with this set to the model instance
sets can fire error events for all models within as well
fix chain test
add tests for and
trigger error even when providing a callback
merge pull request from garrettmaring master
merge pull request from braddunbar
cleanup reverse line
merge pull request from braddunbar url param
revert fixes ensuring that sync is triggered on the collection after the model is remove
add a missing var declaration inside handleevents
merge pull request from cdmwebs master
merge pull request from braddunbar add duplicate
addinga a test case for
trigger with object behaves dumbly
fix escapehtml function
cleanup collection update
fix up global in view test
further isolate remove by exposing removeelement in case subclasses forgot to clean up
elaborate on history decodefragment
merge pull request from lennym master
merge pull request from braddunbar history
fix edge version incompatibility
revert issue number in comments
remove unused var
fix recursion error when saving from within a change event
bail from listento early
add ability to avoid an additional call in setelement method if current element is already wrapped with
fix for issue
merge pull request from braddunbar decode fragment
merge pull request from braddunbar ctor args
fix setting of id for todo model
assert that change is only triggered once
pass options to success function on save create and destroy
set correct number of assertions
remove current check from checkurl to allow loading of empty fragments
add test coverage for route precedence as per issue
fix namespace reference in the todos example app docs
add append option when fetching a collection useful when lazy loading data sets
no periods on errors
merge pull request from jridgewell model id test
tweak parse opt to be parse attrs for descriptiveness
remove redundant attributes check
fix nested change events trigger with correct options
pass along model and comparator to cloned collection fixes
slight refactor to old ie detection
allow collection at to accept negative indexes
add failing test for save and wait
add attributes parameter to backbone view all attributes are being applied to view s this el during element creation
add failing merge with defaults test
merge pull request from gsamokovarov model change test
fix issue in patch not sending data
clone options passed to reset
add correct number of expected assertions to nested change events model test
merge pull request from braddunbar wraperror
fix error message on missing event callback
remove dead code
making the speed test labels fit on the graph
fix the parse inconsistency in the original mmanner
optimize regular expressions the capture groups aren t needed used the character classes will do the job just fine
merge pull request from gravof master
allowing view events to be functions instead of simply names of properties on a view
merge pull request from braddunbar trailing root
don t bother with collection set unless models
merge pull request from rgieseke master
make it possible to take advantage of jquery deferred with backbone without breaking compatibility by changing return values
allow view el to be provided as a function
patching a pernicious pushstate problem
tweak and merge
fix issue with atroot matching
router test tweak to placate oldies
test render returns view
return the result of listento in listentoonce
simplifying haschanged if statement
fixes allow view events to be defined as a function making it easier to call super
merge pull request from tomasztunik master
merge pull request from braddunbar old ie tests
merge pull request from smelnikov fixspell
events off uses reverse iteration to avoid altering indicies
remove undelegateevents call from remove
remove extraneous hasstate flag
add test to show listento called on the same object
fix clarify eventsapi logic
update backbone js comment for clarity
merge pull request from aeosynth master
use on in all cases of listento
merge pull request from braddunbar at root
fix merge conflict
allow the passing of any number of arguments to model initialize
fix incorrect variable name on docs for delegateevents
pass the error and success callbacks when creating the model to add to a collection
check documentmode and combine checks
merge pull request from braddunbar contains
unit test for ignore values when unsetting
merge pull request from braddunbar root
fixes fixes allow routes to be defined anonymously
fix earlier commit got my defaults ordering backwards
allow collection at to accept negative indexes
allow collection to receive falsy comparator to override default
much asked for change instead of throwing an early error when add duplicate models backbone will simply skip them instead
merge pull request from vovik new master
merge pull request from devinhunt master
ajax processdata is left as the default if emulatejson is set to true
merge pull request from braddunbar inline add remove
add a test for
merge pull request from braddunbar trigger args loop
pass collection fetch options along to refresh
fix documentation and tests for negative ids
testing expected behavior when model parse returns null
no reason jquery can t be a little faster too
do not update models if nothing is set
fix failing test cases
fixes allowing unset to work with changedattributes
merge pull request from braddunbar search test
merge pull request from keithcirkel master
use a while loop instead of do while
encodeuricomponent this id
return this from listento
return this reference
merge pull request from carldanley master
merge pull request from michalkot master
add test for once without callback
add backbone collection tojson returns array of model s tojson
use a simple slice
clean up references on both listener and listened objects
match save to new set signature
merge pull request from crisfole update underscore
tweak id classname test
pass options through to tojson
use ok false instead of non existent fail
only remove single when getting url fragment
simpler arg parsing
cleaned up the inherits function and add some extra comments to make it clear what is happening
add test for sorting during collection add
failing spec for listentoonce
merge pull request from gsamokovarov refactor events test
remove duplicated context
mild dry ing and fix logic error
add missing word to emulatehttp comment
fixes create triggering an improper parse
create iframe conditional on options and hashchange and pushstate support
tiny tweak to otherwise it s looking pretty good
revert skip pushstate tests with file protocol
add listento yourself and cleaning with stoplistening test
merge pull request from braddunbar once noop
merge pull request from braddunbar style
fix for consistently use attrs in model constructor
merge pull request from jdalton ie
merge pull request from lukeasrodgers use raises in tests
fix style of previous merge
return this from view setelement
giving collections a clone method
fixes updating comment to reflect changed style of validate
shorter defaults in model ctor shorter extends in history
ensure a search test fails when appropriate
pass options to the coll add call in collection create so that it can work silently
remove initialize defaults logic defaults handles this for us
return the jquery object for el when this is called without argument
test for unset attributes not being returned by changedattributes
another mootools remmnant
use hash literal directly
remove accidental globals need to get lint hooked up
fix add a comment explaining performance concerns in trigger
add missing comma
merge remote tracking branch wyuenho view native hooks into view native hooks
fix set location hash with a leading
update due to recomendations
better method names
make protoprops optional when inheriting
merge pull request from jisaacks master
fixes inconsistency with constructor initialize
add options to the onerror direct callback
add test for
add a comment about iframe history
fix backbone ajax implementation
accounting for event maps
refactored to replace comparator string with the generated function instead of generating on every sort call
merge pull request from braddunbar trailing space
merge pull request from braddunbar simple
merge pull request from braddunbar implicit tojson
always trigger error when validation fails
merge pull request from megawac lto ret
remove unnecessary assignment
working version wip
merge pull request from braddunbar issue comments
merge pull request from braddunbar parse test
use collection in collection fetch
trigger error on originalmodel not model
simplified falsy comparator test case
fix for unset attributes not being returned by changedattributes
fix flawed test
tweak isobject instead of typeof
move sort after merge
fix collection push should not sort
fix change event should be fired if a change occurs
add failing test case for different idattribute fields
check if obj is model in collection get
test for correct context in event maps
merge pull request from braddunbar collection model
merge pull request from asan master
dry ing up setelement a bit
remove a test that is invalid in ie
remove superfluous spaces from model test
modelproperty doesn t need to check for null
style coll collection
using jsonp since the same origin policy won t let file make an xhr to github com
fix syntax error
use surrogate instead of prototype
match comparator string checks in sort and add
add tests for
fix collection create success arguments
fix a small error in the documentation
add a failing test for collection create not running callbacks
fix error when add model to collection at an out of bounds index
merge pull request from josscrowcroft master
fix update todo example with new success callback signature
exit early in collection create
cleanup reset by using add to extend merge and remove args
return this from delegate
merge pull request from dillonforrest fix descriptive parse tests
partial issue always pass options to error callbacks events
make item arrayizing consistent through the file
fixes don t return the parent constructor s value
merge pull request from ymichael master
test null set doesn t clear model
nicer jq test
fixes ensuring that sync is triggered on the collection after the model is remove
issue number in comments
add a comment about iframe history
fix error callback in localstorage sync
fix two parse tests to actually have something to parse
fix variable scope error provide test
backbone extends events
merge pull request from tgriesser patch bug
remove extra guard
collection add should not alter models array
merge pull request from jridgewell listentoonce noop
merge pull request from braddunbar el function
add test for urls in params
model constructor only take attrs not models
fixes edge case for titanium
merge pull request from wyuenho view native hooks
check for onpopstate in window instead of window onpopstate
fix typo in once comment
merging in unsetting id
decode location pathname
merge pull request from wettowelreactor fix
allowing collections to remove models by either cid or id prefering cid
merge pull request from wookiehangover whitespace
revert non model sync commits
use notequal body isntead of equal null
add a third context argument to backbone event bind which when passed will bind the callback to the provided context
don t just test for the presence of options test that it points to the same object as passed in
make router routes accept a function
fix indention in tests
throw an error if backbone history start is called twice
avoid throwing when is not defined
merge pull request from jdalton coerce
fix tests for ie webkit regex not string match
refactor to more better fix and add some more tests
off wrapping callback not the original
multiple views may listen for events on the same element
merge pull request from jridgewell history use
merge pull request from braddunbar history start
made backbone sync options parameter optional
add tests for merge true
reduced extends to single return
fix lingering idattribute spot thanks jdknezek
very minor code formatting
merge pull request from braddunbar speed test
merge pull request from ganmor sortfix
remove redundant assignment in events trigger
fix small leak introduced by
return this from delegate
merge pull request from tbranyen silent start
ensure listentoonce without a callback noops
updating model previous arguments check to match haschanged
fix update viewoptions comment
route parameters provided as array argument
sort option for add
add extend to backbone history
merge pull request from braddunbar phantom pathname
remove double event binding simply stashing the original callback works as intended
test for conflicting classname id and attributes
ignore values when unsetting
be sure to return the value of beforesend
test complex query strings fails in some browsers
refactor var name list events for clarity uniformity
simplifying trigger condition
revert fix whitespace consistency for
fix typo in test for presumably equality
remove speed test from unit test suite
remove unneeded assignment
fixes history is started before navigate
merge pull request from rstankov view dispose can work with normal objects
remove unnecessary variable matched
save a remove loop by setting remove to false in reset
initial support for models with non default id attribute names mongodb couchdb various tickets
simplify test for backbone events inheritance
add test for
merge pull request from colevscode master
fix collection set regression when parse returns falsy value
fix assertion count
merge pull request from ottoallmendinger master
merging in willmoffat s patch
merge pull request from trismcc master
fix comments typo
add a route test for unicode paths
merge pull request from emorikawa master
merge pull request from jerryorr master
don t blow up if you try to stoplistening an object but have never listened in the first place
allow history start to not call initial route
allowing a collection to be passed into a model via the options
fix create default backbone history
add trigger applyment
add comment remove anchors
call beforesend function set with ajaxsetup when backbone emulatehttp is set to true
tweaking transparent url upgrades
missed an api regression in model changedattributes reimplementing in the same way as the original
cleanup model save success callback
add test case for root with mandatory trailing slash
fix localstorage link
merge pull request from captbaritone indent fix
fix wrong test collection model must be a function
fixes by cleaning up a messy conditional
code style tweak
merge pull request from braddunbar history window
lint style tweaks to prev commit
fix use apply to implement slice
test undelegateevents returns view
add view remove
set the pathname explicitly for phantomjs
merge pull request from braddunbar isvalid
merge pull request from braddunbar falsey haschanged
move collection assignment a few lines up so this collection is available in model parse
remove test elements after insterting to fix failing global tests
merge pull request from jridgewell collection removemodels
test case for model id
merge pull request from vovik master
get escape in the docs
pass usecapture to removeeventlistener
style tweak remove jquery dependency
update collection js
slightly saner validation on initialize
merge pull request from braddunbar ajax
merging issue others validation on collection create
merge pull request from reconbot master
merge pull request from tgriesser localstorage fix
use qunit raises instead of throws
bitwise declared bitdumb revisions ensued
do not call loadurl when redirecting to hash based url on non push state browser
merge pull request from sophietk collection predicates fix
add test for negative indexes in at method
merge pull request from nhajratw master
correct minor mistake in todo example
merge pull request from hartzler remove unreachable code
be explicit about boolean trigger option
default constructor should return parent value
refactor account for double parse bug in update
merge pull request from wuzhe upstream master
add a bit more of a test for changedattributes
merging in hash fixer for ff
merge pull request from lmp master
use clearer event names
check typeof length instead of instanceof
merge pull request from megawac where
remove erroneous console log
merge conflict resolved
fix named and optional params
fixes assign collection in preparemodel
ungolf create assignment
add a test for issue
router test tweak to placate oldies
don t send up model data for get requests by default for folks who are re fetching existing collections
on is a noop if no callback is provided
merge pull request from braddunbar route names
fix add backbone view destroy
fixes default to id if there s funny business going on with a missing idattribute
disallow chaining in hook calls
fix spelling error in documentation
enable qunit config noglobals in tests
merge pull request from djebbz master
using isequal to check model change
test remove returns view
add tests for
merge pull request from gsamokovarov todo difference
add a test case for cloning the attribute object
changed trigger call
tweak comment a bit more
avoid creation of extra array in collection proxy
fixes enhances backbone events to support jquery style event maps in on and off
no need for repeated comment on remove
merging in a test for issue
merge pull request from jridgewell collection removemodels
use single quotes and drop self closing slash
fixes load initial models after collection initialize not before
merge pull request from wrapp master
passing falsey keys to haschanged or previous
hopefully simplifying and optimizing trigger further
ensure attr is a string when passed to this map
fix object notation for events once
fixes model save null wait true with validation
fix licensing typo
merge pull request from hmil fix options success
fix collection create success callback passing the model parameter through
merge pull request from lowkaseo listeners rename
enhancing model with isempty function
issue unsetting a missing attribute should not fire a change event
allow non model object as second argument with test
remove xhr polling restriction
merge pull request from braddunbar consistency
add failing test for
merge pull request from jridgewell variable cleanup
merge pull request from caseywebdev global events
merge pull request from adictovirtual master
fix broken comparison
fix correct comments for fetch
add a more meaningfull test for parsing by default
merge pull request from cdavaz master
issue remove the optional bang from hashstrip too many inconsistencies
stubbing out view test
merge pull request from captbaritone nested ternary
remove unused arguments
taking some of dvv s suggestions
support merging of updated object state on collection add
clarify reference to the parent constructor
add asynchronous test for once
shorter delegate method thanks to jquery empty selector checks
add tests for
replace non standard substr with slice
fix unit test for model isnew
tojson is called implicitly
merge pull request from jridgewell model url
merge pull request from caseywebdev update
fix to pass with edge version of underscore
defaulting backbone collection model to be backbone model
merge pull request from jridgewell trigger objects
merge pull request from braddunbar transition to hashchange
fix leaking global
tidy up event internals delete empty event keys in off return this early in stoplistening
add events onetimelistento
merging dxgriffiths annotations to inherits
merge pull request from braddunbar no bindall
have reset specify merge false to update
merging issue multiple views per dom element
merge pull request from braddunbar trigger multiple
fixes make route chainable
skip pushstate tests with file protocol
fix test case
updated issue number in test to match pull request issue
pass along the current value of this to backbone s closure wrapper
failing test case
update backbone js
merge pull request from jridgewell test view
remove lingering comments on silent changes
fix test making vars local
remove jquery specific selector
merge pull request from braddunbar default options
merge pull request from tgriesser create model fix
merge pull request from braddunbar url root
merge pull request from caseywebdev fallback get
update test to fail
merge pull request from rutkovsky master
consistent argument names across event functions remove unused argument from triggerevents
implement once with minimal change to other events methods
attributes should remain a safe copy
merge pull request from gsamokovarov listen to cleans itself
period in comment
update backbone js
model url id falls back to get this idattribute
send add and remove events through the model
don t be clever
remove null options guard
improve test merge in duplicate models with merge true
add test for
whitespace cleanup for extend collection prototype chain function definitions
more descriptive variable name
simplify conditional in collection update
use matches in where
fixes don t bother trying to load jquery on the server side
add modernizr style check for onhashchange to avoid issues in ie compat modes closes
haschanged set should use the same comparison
merge pull request from mickeyreiss master
fixes previous commit
merge pull request from caseywebdev set refactor
use collection where for todolist remaining
add test case for trailing slashs changes to atroot in backbone history start
no need to default options to
avoid nested ternary
remove accidental inclusion
merge pull request from braddunbar haschanged comparison
setting to context to this by default on bind instead of trigger
merge pull request from shesek master
slight optimization for events cache splitting regex
allow empty routes
add test for
merge pull request from braddunbar simple
use empty route name for anonymous routes
