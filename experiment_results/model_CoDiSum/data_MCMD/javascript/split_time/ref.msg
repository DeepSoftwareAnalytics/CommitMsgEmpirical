use isdirectory instead of mustbedirectory
injectable browserwindow constructor in atomwindow
respect headless on non spec windows
initialpaths and representeddirectorypaths renaming
guarantee the window opening order
allow launchscenario launch to accept an existing app
stub the storage folder application json read
don t require urlstoopen
don t append launch options to the deserialized ones
don t restore state when new window is passed and other paths
test for window restoration on launch not open
revert accidental change
don t select search result unless user does something
add preservefocus flag to keep main process tests from stealing focus
test behavior for collapsing new paths into restored ones
save projectroots not initialprojectroots which is undefined
did change paths is never broadcast this event is handled by atomwindow setprojectroots now
save window options when opening a window
left in a console eyes
use newer oses for tests
feat mp alipay 兼容 request 方法
update constants js
fix components 去掉了tabbar中的array prototype find
move foreachdroppedorpastedurl util to src
don t reorder a single child
no need to reorder empty children either
return an all promise for windows opened by urlstoopen
return a promise that resolves once the server has started listening
default devmode safemode and window to predictable false values
consistently return an atomwindow from openpackageurihandler
don t require locationstoopen in load settings
stub trivial isspecwindow method
url pathnames use not path sep
call removewindow explicitly to simulate a window close
line separators are hard okay
fix lint errors
distinct test names
consistently operate on result pathtoopen
memo document the launchscenario stuff
use hasopenfiles to flag a window that will receive openlocations
clarify behavior in update blueprint action
test for restoring windows over directories and files
clear dom overlay on shutdown
update provider js
ignore leading edge of find by dom throttle
fix messed up rebase
add jsdoc for replacenode parameter
fix don t customize esm options options like cjs resets other defaults this ensures consitency of usage release
fix esm test
docs for scene events
remove baseplugin boot
docs for baseplugin differentiate the nonshared properties
merge branch master of github com airbnb lottie web
add more specific sub command error handling
file uploader instead of uppy dashboard cc kvz
refactor eslint 取消掉无法在类函数中使用 jsx 的规则
remove сюда here
enable compilation of tools static assets server
update spec main process atom application test js co authored by smashwilson smashwilson github com
don t need weakmap for properties of same type
run first setstate with explicit priority in case of non react event handlers
fix wrong import
fix destroy tray when app exit
update json reporter
revert null check removal
feat add config schema valid
refactor check pid is running before kill it
add setclassname method
fix update per review comments
fix extra semicolons in german locale file
refactor eslint 取消无法使用函数式组件的 eslint 规则
feat cli quickapp 快应用样式尺寸处理
update russian strings drophint poweredby authenticatewithtitle
feat quickapp 快应用事件处理
fix quickapp 快应用属性传递最后小写
fix quickapp 修复快应用路由相关 api url 处理错误的问题
art formatting tweak
fix redux 快应用属于多页应用 数据改成挂载全局下进行共享
document why open file is there
fix flow error
update index js
don t show informer if it s an auth error
update src main process atom application js co authored by smashwilson smashwilson github com
re add the selection message
feat remove similar unuse config key
remove trailing space
updated by alf automation
add customhitarea boolean property
fix bug with deleting members after config change
set the panel flag as focused when activate is called
transloadit add assembly status property to assembly errors
keep the property but set to null by default
transloadit add opt out of error reporting
transloadit remove ip report full error message
fix null is not an object error with currenttarget instead of target
delete stale paths
add support for option
add missing translations
fix line styling dashed thick when not using html labels option
use companionurl in react native example
atomic children removal
updated by alf automation
set newwindow true for options recovered from application json
test version ensure prompt default preserves prereleaseid
chore fix simpleset export
don t disable the vertex attribute if the location is invalid
add quadshader pipeline to the renderer
updated by alf automation
updated by alf automation
fix last tag detection
y oh y
add transformonly flag
remove unneeded variable
companion assume higher size when size is uncertain
hot fix for crazy totalprogress cc goto bus stop
use now for spawnchildprocess because of upgraded machine runner in the new version
update usage of spawnchildprocess in test
fix map disallow setting map key with special properties
fix page count computation in publicsuffixlist enablewasm
test update localizedformat test
compress diffchildren further
compress diffchildren further
make the german taglines a little shorter
merge branch master of github com transloadit uppy
changed owners back button label to say up to
add getduration and getdurationms
use proper elapsed time for the shader
fix mo protocol new bt task
test eslint 修复升级 eslint 之后的报错
fix typo recursivelyiniitliazetree recursivelyinitializetree
commit improved code formatting suggestion co authored by tiarhai garyedwardthomas yahoo ie
update retrieveregistrationstatus js
main process test for dev mode and benchmark windows
update api js
fix taro qq 修复 qq 轻应用 hooks 依赖路径
replace deprecated del with delete call
fi mp alipay 修复支付参数不正确的问题
fix taro qq 修复 qq 没有给组件实例加 prefix 的问题
re apply stencil after a bitmap mask
fix issue by marking dirty before render since this can potentially add another entry on the render queue
add back mistakenly remove jsdoc comment
fix graphql type json has changed its commonjs export interface
expose parsexmlbitmapfont for combined xml font file
workflow publish version marker after all other packages are published ci skip
workflow fix release script
feat allow installing core plugins from next dist tag
workflow fix vue cli version marker version bump for prereleases
flag which version has this method
workflow fix syncdeps for prerelease versions
test fix assertversion test
test fix assertversion test for prereleases
test skip version comparison test for now
note re programmatic usage of generators
optional texture key array
better default checking
constraints canvas size if using fixedwidth fixedheight style
remove redundant code
remove redundant code
constrains canvas sizer via game object size
do not wait until config watcher is ready in order to launch the atom window
try catch the uniform json parse
fix a path selection bug when filter pereferences are updated
fix missing verifyinjection renderer for engine test
fix position of module doc for engine
only delay the watcher on windows
add more explicit comment about windows perf issue
explain why we don t use randombytes on a comment
log every require done
switch to self rather than window
fix lint error
fix lint errors
shim ondiderror method on returned watcher need to figure out what to do with this for the long term
react native example fix chunked reading cc kvz arturi
increase arbitrary sigkill timeout to make self test more robust
ignore dangerouslysetinnerhtml during hydration
try to get azure devops to run
fix broken path selection after filter changes
can we directly require browserwindow
figure this exception out
avoid duplicate plugin issue w sfe
add option to disable sfe
check for version of node that the sfeplugin requires too
increase the timeout for the filerecoveryservice tests
replace setimmediate with settimeout
chore remove extraneous log statement
remove un used properties
only clear profiling snapshot on root unmount
dom element will check if parent can render
fix tests due to escaped chars in comparison string
properly fix escaping in test comparison
docs add fixuserconfig comment
fix issue when disabled option was highlighted on the active state
add active to styledsingle and styledmulti docs examples
add it in front of spec descriptions
compress diff code
replace map with some get rid of intermediate array
compress code in diffchildren
don t explicitly initialize isnew in diff to false undefined is falsy anyway
compress diff code by reusing a local temporary variable
reorganize diffprops a bit to compress code
warn about missing preact render to string dependency
rename t to error in createsuspension
fix exports in compat
drop console from devtools demo
keep displayname of lazy static
add lost semicolon
fix progress bar animation remove foreach not needed currently
fix a typo in query prototype distinct docstring
remove outdated todo comment
merge branch master of github com automattic mongoose
style fix lint
ref to next instead of latest
use lazy displayname instead of name in debug
fix eslint 使用 commonjs 模块化
fix atom dev with json config files
scroll line to center when opening file at specific line
fix double doc entry
update test to expect new rendering of missing tree sitter nodes
vim add support for text object
fix fix draw function can only call once
refactor taro 使用 object is 判断 hooks 的 deps 是否相等 与 react 更为一致
feat restore window state at init
style fix lint
fix atom open
try running with verbose
okay add it to the end
pray tell what is our arch
fix a deprecated prop warning from react window
art debug message
log apm install output
fix rn toast 或 loading 不对称情况下防止泄漏
companion fix test
fuzzy match in the should parse postscript numbers unit test to work around rounding bugs in chromium browsers
set apigateway variable default to an empty object
remove unnecessary checks
remove other unnecessary checks
fix ensure signal is sent to exit event
rename test for additional clarity hopefully
always world step once
skip test that will be fix later
feat don t compare when there is no olddom present
fix debug message
chore undo change and remove only
update pathwatcher fork to fix windows build errors
chore improve naming
reduce pr noise
fix self is not available in non browser env
use published fork meteorjs pathwatcher instead of github tarball url
default to native watcher on the stable channel just to be safe
fix the lint error
add missing plugins jsdoc hook
restrict deep imports
fix mp weixin 处理 previewimage 不支持的参数
add debug test with throwing loader in lazy
switch to ownerdocument defaultview
fix merge typo
fix changelog message parsing with additional
add root types namespace
physics root type
add a hook for plugins to perform asyncronous initialization needed for sfe to enable deployment profiles
jsdoc fix returned object type
fire unused methods
remove only from test
catch realpath errors
count size past the buffer limit
fix discriminator function parameters doc
not logging warn when missing character is a new line
fix taro swan 百度小程序加上 diff
chore fix typo
test custom restapiid resolution
pretty witty fine
add the modes back in as we need them internally
switches to https for dev testing
merge branch master into master
fix clean path
use kexec for mac linux springboarding again
update test to fix ci
normalize to realpath in waitforevents helper
chore helpers disable overzealous lint rule
missing grid properties and fr unit
tests improve tests
feat add poc solution
remove poc solution
fix components 修复 swiper 下标指向问题
fix exifreader typo
use the todoelectronissue keyword to make the comment easier to find
make linter compatible with windows
add script lint fix which fixes some code formatting issues via eslint
add import statement at the beginning of file
fix lint error
do not use arguments as variable name
test fix tests
update tidelift message
chore fix typo in comment
update compat src index js co authored by cristian bote cristianbote users noreply github com
fix github commit did strange things
add type application javascript when initializing worker
corrects path logging
improve error message
accept custom restapiid only if in directo form
load hack only if meant to be invoked
improve test title
relax rest api validation
fix typos in error message
improve tests reliabilty in real world state does not equal serverless
fixes unsupported images like png or images without orientation
ensure to delete env vars due to string coercion assigining undefined sets them with undefined string
improve mock setup
fix restoration of serverless env
ensure no side effects
cleanup process env handling
ensure test doesn t depend on env state
fix lint issue
fix popover arrow by selecting the first element child
fix only add each process name to regexlist once
add to markdown quotes
test config ensure clean install test aka npm cit uses clean jest cache
refactor run topologically do not pass figgy config down to constructors directly
fix rn navigationstyle 配置失效的bug
fix selection jumping after long tap on ios
support for sfe outputs in config
ensure needed test methods
ensure needed tooling in test
ensure needed tooling in test
ensure needed tooling for test
ensure needed tooling for test
ensure needed tooling for tests
ensure needed tooling for test
disable atom notify again
throw when dist target does not match transformed entry
update meteor babel and reify versions in dev bundle
use reify to compile dynamic import
mimetypes js add pdf txt mime types
always flushpassiveeffects before rendering i mistakenly wrapped this in the revertpassiveeffectschange feature flag it should flush regardless of the flag
ensure to not wait for stdin when testing
ensure we do not wait of stdin input in test run
xhr upload set upload protocol
companion validate upload protocol
ensure to cleanup temporary files
upgrade glob pattern
remove js from js map glob
feat add seed time to system config
tests ignore unreachable line
merge branch master of github com automattic mongoose
flip the test for dangerouslysetinnerhtml
fix 解决 safari 浏览器报错的问题
use the correct ripgrep path on asar packages
fix undefined chart node when switching commits in profiler
companion client provider views send uppy client header query
fix typo namesapced
companion client remove version header to avoid block headers against older companion
providers change versions header
companion add todo
disable failing ios test for now cc goto bus stop
improve test mock
be more explicit about nightly tag target
cancel previous build at start of new build
improve check for graph validity
bump sleep in flaky server test
docs schema minor improvement to pre and post api docs
return activatepromise when calling packagemanager prototype activate
remove arrow function syntax
fix add selective torrent list utils
exit with non zero status if there were failures
ensure to return promise
ensure to return promise
ensure to return promise
configure detection of orphaned async calls
webcam fix supportsmediarecorder test arrow functions do not have a prototype
ensure to return after promises resolve
ensure to return promise
ensure to not leave orphaned async call
merge branch fix tests setup into improve promised bdd setup
improve whitespace handling
fix issue where syncslide would not run play newly add background content
update args logging
expose the matter body module
drop only on suspense test js
remove console logs
fix path issues in windows
ensure to not send track requests when testing
fix isjquery check to work with latest chai
remove sfe node version check
remove sfe node version check change what require hacks we use
fix failing tests related to chai upgrade
style fix lint
minor changelog refinements
eslint auto fixes
merge remote tracking branch origin develop into release candidate
chore eslint 修复测试用例的报错信息
ensure to return promise
speedup unhandled rejection notifications it s to ensure it doesn t interfere with async leaks detection
ensure to return promise
ensure to return promise
ensure to not send track requests when testing
ensure to return promise
ensure to send user tracking requests
ensure to return promise
ensure to return promise
resolve promise on close event as this one confirms closure of all file descriptors and works with async leaks detection
increase timeout as test occasionally times out
ensure to not issue track requests when testing
ensure to not issue tracking requests when testing
ensure to issue track requests when testing
uploadcdn update comments
changed export code to hopefully support larger exports
add datetime to the profiling export filename
jsdoc make createdynamiclayer position optional
server clear error on success
re enable cli reporter
accept a number in cli reporter gettext
fix a missing remove event listener in profilerstore
export import unmount
unescape slashes in regexps
doc release show objectproxy content as public also updates docs to proxying to arbitrary unknown values
update electron builtins
companion set contenttype
add tab id to tooltip
refactor new bundle creation
increase timeout as timeout crash was observed
fix coding style
ensure no false positives on async leaks check
ensure to return promise
remove inputqueue config option
tweaked event display name logic
tweaked bridge shutdown sequence slightly
handle https git clone urls
skip register tests for now
fix component 修复一处typo
log it all
game gettime would return nan because it incorrectly accessed the time value from the timestep
prolong timeout test failed the ci with timeout error
whole lot of debug code
rfc don t read attributes during hydration this would represent a real size performance improvement
remove unneeded eslint directive
re add enterprise disable d sflag support
fix plugin resolution tests
fix unrelated test
add donate link to website
fix move playground link back
looks like there are windows being returned
perhaps it s something with await
add csound includestr directive
fix server return listener when calling listen
dump hocr tsv box unlv osd in final result
i think we need to pass in a folder
pass in filepath as an argument
atom workspace should always exist
prevent extra space in changelog
delete the session when we re done just to be safe
companion more comments
add more variation in autoinject test
avoid exception when highlighting empty code blocks
run prettier on changeset config and make publishoptions public true
change another react select import
remove unused function
revert a change
first last slide shortcut now uses shift modified cmd conflicted with browser back fwd
fix async and creatable docs
use ternary instead of conditional
fixes linting errors
execsync test util
execsync test util
log output in case of error
fix discify script now that we have json files in build
really check for release commit
ensure everything is rebuilt after version bump
update after version bump description
save some bytes
correct types for groupcreateconfig key groupcreateconfig frame
docs description for groupcreateconfig
add force f flag when remove service
remove garbage characters
uppy drag drop made onbeforefileadded be called twice even if file is the same
fix suspense exports
do not allow setting invalid values to tabsize
fix default enable check auto update on macos
don t clip if path is undefined in svg back end
catch poll zero
remove touch limit on over out
duration is optional
add numbertweenbuilderconfig to path configs
fix lint issues
companion client add version properties to requestclient provider
increase timeout to avoid false positives
merge addon fix typo in hasmarker this was causing exceptions to be thrown when using custom markers
ensure eslint treats modules as cjs not esm
fix eslint configuration of strict option
document disabling of rule
configure import no extraneous dependencies rule
improve tests timeout in face of failures
linear navigation fix showing arrows when last slide has vertical children
fix lint errors
extra properties for tween configs
add the remaining tween config properties
fix core test teardown helper
add note about failing core test
enable all non core tests
remove unused exports
events fix tests for alreadydisposederror
backed out bridge shutdown refactor as it seems to have introduced bugs
update packages uppy companion src server uploader js fix linter co authored by artur paikin artur arturpaikin com
update mp mp alipay 放开tabbar api 参数不需要转换
fix lint warnings
ensure to cleanup temporary files
merge branch ensure cleanup into improve env vars handling
ensure to not silence uncaught exceptions
ensure to revert current directory change
make cachedir always an absolute path
move require to top it seems there s no reason to have it nested
expose tmp dir common path
ensure to mock homedir for tests
fix lint issues
ensure to not respect cli argumenst for sls config
fix lint issues
expose process output in case of crash
correct ease type
remove a newline
ensure to pass home in test isolated script
ensure to mirror mandatory windows env var
update website inject js
update website inject js
fix 修复rn中 页面开启enablepulldownrefresh后 页面下拉刷新失效
omit prefix to avoid doubling up when key gets set
add matter bodies
ensure to test non colored output
docs correct search for tutorials
native crx fix dispatch events nw
fixup native crx binding support
disable guide group highlight in owners tree mode
test fix new line
fix rn toast api 错误回调
fix a typo in string truncation util
fix box drawing chars comment
add hoc badge to owners list dropdown
disable guess js for now to get site to build
fix linting error
back to manual debugging it is
update web js
sync function stub may throw not reject
merge branch fix async stubs into upgrade sinon
update sortable js fix bug fix error in angular server side render
update to new version of sinon
npm run build service legacy
fix equality test
add regression test
docs model query link to findoneandupdate docs
upgrade after chai update
update after chai upgrade
ensure needed utils
reset max indentation when entering exiting the owner tree
remove outdated comment and eslint rule ignore
simplified background highlight style now that corner radius has been remove from selected highlight
docs for tween
add missing utils
increase timeout for randomly failing test
ensure to expose tmpdir to nested processes
ensure needed utilities
ensure needed utils
ensure needed utils
fix lint issue
fix lint issue
ensure windows mandatory env vars
improve test timeout setting
skip test on windows when no admin rights
increase test timeout
increase test timeout
do not crash if tmp dir delete fails on win
increase test timeout
ignore failed attempt to remove temporary home dir
bring back reset of user config by mistake it was remove with reset of homedir
use more accurate ensuredirsync
link mocha bug report
improve timeout settings
add group name
update motion automatically
fix the issue that code blocks in markdown are rendered as placeholder after disabling nunjucks
use defaultstrokewidth in drawdebug
remove duplicate graph traversal
support pass of after callback
validate against invalid suite injection
support aftercallback in symlink fail skip
shift timeout setting
update web js
ensure proper context
docs model fix typo projetion to projection
use as the directory for ripgrep to scan this misteriously solves issues in windows
new input const and changed order to be numeric order
re add plugin transform react jsx source plugin for tests
merge branch master of https github com photonstorm phaser
fix components 修复 input 类型判断问题
fix linter errors
fix popup bug
add new arguments to resize event
feat add check task is seeder util
fix uploadcdn for local build
fix only hydrate when mounting
updated reload and profile temp key to use sessiostorage instead of localstorage
docs for easedvalueemit
remove dead code
update unmount comment
fix an increment bug in sidebar for fiber commits
test fix tests
test fix tests
only show rendered at if selected fiber committed at least once
add an explicit did not render label
fix chrome headless running on wsl
clike mode remove float as java keyword
correct docs of arcade physics body member
chore cli 修复 cli 发布检验脚本
fix reference in test file
feat add protocols to user config
chore remove print
update dom log to match master
fix some tests update function
update components test js
use minimist directly to parse the config option early rather than instantiating cli
throw an error if sls config specifed doesnt exist
fix unit tests for packageservice
fix use of getserverlessconfigfilepath in packaging
improve memoization key for getserverlessconfigfile
fix minor border bug in profiling tab toolbar
update fragment tests todos
remove console log
bundlereport import type instead of import
don t change signature of loadservicefileparam
use path basename
refactor check is exist before delete task file
resolved an name change issue fom a merge
reference custom apigateway for models and request validators if configured
do not store null buffers on the cachestore
mode metadata add meta entry for postgresq
require truthy random value for randomness
remove extra function calls in the instagram api example
fix missing await in the foursquare api example
style fix lint
update displayoriginx displayoriginy after size changed
update bin upload to cdn js
moved the pick function to the object utils
seclude canonicalfunctionnameprefix var
fix setup of iam policies for functions with custom names
add code comments
fix owner tree for legacy backend
fix duplicate packaging issue
fix typo on mindate test description
changed failed image bg to transparent
improve toequalnode null assertion
doc fix docs for schedule
chore schema reuse mappath variable
support buffer in image of tesseractworker support buffer in image of tesseractworker
fix timeout response template
allow console info calls in tests it s to report on long going tasks or valuable debug information
log child process output in case of fail
disable flaky spec on windows until we can identify a proper fix
make config test flow safe
install apm using ci
fiex 修复 ontouchmove 无法调用的问题
fix scrollbar not being shown on safari
improve user mesage
return whole service config not just service name
improve code comments
improve comment placement
improve comment wording
feat add parseheader util for restart task dialog
remove old debug console log
fix lint error
chore print full stack trace when building docs
chore add markdown filter to all render calls
docs model fix wrong memberof for inspect
improve devtools isroot check
go with default include approach
fix cross browser style assertion
feat empty addtaskoptions when task add
fix there is no localhost rule in etc hosts
style fix lint
refactor auto save session when save preferences
docs index add maxtimems
refactor removetaskrecord error complete remove
eslint config add missing unsafe lifecycle methods to react sort comp rule
increase width of some rows
sort by app name
increase width of some rows
load language rust at snapshot generation time
implement compare on nullhighlightiterator
move lazycomp def into only test it is used in
hint addon handle scrolling containers
remove unused code
don t try to convert absolute paths to relative
fix generation of release notes for nightly builds
add new transform getparentrotation method
add dragstartxglobal and dragstartyglobal properties
make stacktrace undefined if err stack undefined
flare update event component displayname
fix exception when multiple language layers are not yet highlighted
remove meaningless conditional the if and else cases do exactly the same thing here no need for the if
use boolean to test truthiness
replace spread with array from
imitate public interface
companion use decodeuri only when value is set
companion companion client remove add logs accordingly
companion client set default values to allowed headers
fix checktaskisseeder util
get the correct time value
pass policyfilename variable to rpm spec template
reverted unintended change
companion client add fallback for preflight failure
bump minor versions of babel compiler ecmascript and modules
nest cacheoptions in scope where used
use naming to get stackname
restore tree sitter in grammar iterator
fix increase the number of egine process restarts
add addgltexture method to save webgl textures directly into the texture manager
opps clear the texture
feat app plus export global uni
fix bug where graphics methods were not renamed
scheduler fix navigator isinputpending call must be called as a method
merge branch master of https github com photonstorm phaser
merge branch master of https github com photonstorm phaser
even moar coverage
fix test actually
docs api fix some href
sprites need to flip the uv if the source is a gltexture
remove console log in classdb
fix comment docks
add platformrelease to data posted to atom io with crashes co authored by jason rudolph jason jasonrudolph com
include architecture in data posted to atom io with crashes
fix lint error
abort filter checks if rendertotexture
chore example use
chore not match non alphabetical characters
chore remove unused lint rule
companion client send correct versions to companion
companion remove deprecated dropbox field
add support for multiple loader headers
add default value
add docs to xhrsettingsobject
better flip handling
add flipy argument
remove redundant typeof set
move all exports to the bottom of the file
fix lint error
code improvement cleartimeouts remove duplciate
new onactive callback
fix some typos
docs document clarify tojson s relationship with json stringify
add deprecation warning for sanitize option
fix missing loop callback
mark ios as not supporting native drag drop
fix linting issues
fix invalid method usage in members authentication no issue getpublicconfig in members authentication was not called correctly
with a pagequery example
fix frontmatter issue
tags to quotes
mirror code with remark transformer
moved to full lodash package
pass options for mdplugins
include default value for decks
dont treat classic frontmatter as content
if no mdx content no tableofcontents
remove need for root options
colocate tests with the files they are testing
use user s loaders on webpacked resources
move scopes loader into loaders
componentwithmdxscope deprecation warning
add html to rss feed
make sure webpack externals get unset
add comment rekick ci deploy
fix crash on error
fix typo in error message
add checks to prevent infinite loops
add caching to gen mdx
remove another import
fix export replacement
more explicit import replacement
don t use default value for scope and components
update all examples to gatsby plugin mdx remove old katex patch
remove commented out tests
popup null check
remove loopcounter value as it was incorrect
update lib plugins aws package compile events apigateway lib hack updatestage js co authored by daniel schep dschep gmail com
remove unneeded check
remove obsolete setup task
build unminified es browser build too
ensure to expose output of failed commands
fix service provider region resolution
add new quantity property to groupcreateconfig
stagger function change
remove commented out code
cleanup cli flow setup
clear no longer used option
clear obsolete eslint instruction
cleanup region and stage option resolution
ensure to expose non errors in informative way
fix render offset issue fix render offset issue fix size issue when setting radius
fix remove useless fallback
chore update getlanguage comments
more accurate types
update staggerbuilder js
fix solve lint error
cli ux when command is errored do not display process list to avoid confusion
temp rm exitsep cluster
do eslint fix
improve serverless mock
fix taro quickapp 修复事件传参绑定
refactor buildrule operator default plus
md files can have skip and only to help debug
move tables to gfm
ensure user only mode for credentials
update staggerconfig js
mark reload and profile attach as configurable
fix deep scan
temp remove write sep
rely on more natural array content check
rely on recommended usage
fix taro quickapp 快应用属性处理
simplify mode access setting
do not tweak file access on windows
fix process is not defined issue
update deprecation warning
strict block only if match is anchored to end of hostname as per feedback from filter list maintainers
ensure all code evaluated by node is compiled with reify
fix link to query prototype select
docs document why we check for the vnode
add back accidentally remove semicolons to escape js
fix avoid doing a render null when we have stacked portals with conditional children
fix 解决 inneraudiocontext 对象读取 buffered 属性报错的问题
chore revert only on portals tests
use pathwatcher again instead of meteorjs pathwatcher fork
remove useless dependency
fix intersectionobserver 对象方法 relativeto relativetoviewport 支持链式调用
merge branch fix into dev
append the release channel to the appusermodelid this allows each release channel to be pinned separately on windows
remove useless code
fix outdated import of optimism package in dynamic import test
save only modified immediate hidden settings
test fix cliserviceversion for generator unit tests
fix window open defuser matching logic has been erronesouly reverse
add missing newline
fix typo in comment
set explicit line height to avoid normal resolving to different values based on text
test bump test timeout for upgrade tests
test gantt spec add parser test for inclusive end dates
fix ganttdb js clear formats and excludes on clear
test ganttdb spec fix typo
test ganttdb spec flip function design to specify inclusive instead
make possible of work shift mouse wheel horizontal scroll
avoid transpiling d ts files when bundling meteor tool
fix test webserver to handle paths correctly on windows
fix support uncontrolled inputs in props diffing
chore add test case thanks marvinhagemeister
chore add checkboxes test
refactor newline the closing brace for readability
fix the test for dangerouslysetinnerhtml hydration
add failing event handler test
refactor ganttdb separate out the inclusive end date toggle
test ganttdb spec fix missing initialization for clear test
chore ganttdb export durationtodate so it can be tested
test ganttdb spec add simple tests for the durationtodate fn
add locked property
add onpointerlockchange handler
fix eslint issue
uppy dashboard connected labels to inputs in filecard js
updated docs and frame source checks
fix aws setup link
expose customized inquirer for other plugins
fix taro rn 修复showloading没有使用title属性
chore test update excerpt test case name
style fix lint
fix taro router rn 修复rn端触发不了下拉刷新事件
add fallback for requestanimationframe for when tab is not focused
fix taro quickapp 快应用获取页面参数时增加容错
update aws setup doc link
remove event hook of children when destroychildren is false
docs model rename conditions filter
companion display truer error during oauth failure
companion client remove redundant url in nested error message
provider views set correct package when sending version
export mongoose cast
chore rn update demo
show available env vars in the help output
fix serialize test
javascript mode fix use of let
merge branch master of https github com photonstorm phaser
docs model clarify that findbyid id param can be any type
docs model link to find by id tutorial
update path js
fix interactivecli injection
remove useless return statement
chore 修正 cli 创建项目验证脚本
corrected manager type
fix fix spelling issue
fix fix some required changes of language short name
fix remove not requirement changes
add new properties
fix to input a number value to props
extract pixel from correct location for a framebuffer
fix circular dep and skip htm test for now
fix aws kotlin nodejs gradle test
test eslint 修复测试报错信息 typo
chore fix prompt message
update scalemanager js
add snapshotcanvas method
temporarily disable sigkill self test on devel branch
remove process exit
log message out when user says no to aws setup
remove unused regex
improve error handling
fix bug when using textarea as input component
fix aws secret access key validation allow chars
improve postinstall message
add rootbone support and remove runtime property no longer needed
fix npm conf otp should default to undefined figgy pudding is very strict
fix publish otp cache should be seeded from conf value not cli directly
support for calling tenant orgs chosing between the two is handled by sfe plugin
don t mirror permissions in browser builds
fix portal children always being mounted
fix stacking portals scenario
gameobjectfactory remove is a new static function that will remove a custom game object factory type
gameobjectcreator remove is a new static function that will remove a custom game object creator type
update spinegameobject js
rename arial label to aria label
actually update the right aws creds link interactive setup aws
set fake creds in request test to try and fix ci
doh credential needs to be non empty
doh actually need to be valid creds
only serialize known object types
ensure bundlereference is registered with the serializer
set dirty flag only when render state is changed
fix taro memo 只需要判断 props
fix isolated test
slightly more simplified dispatching of the findbarclose events in firefoxcom js
correctly stub userstats track
improve components setup
improve placement for components injection
relocate stack trace limit setup
do not do forced exit
do not do forced process exits
remove rudimentary exit code setting
remove not used endpoint
clear obsolete comment
seclude const to top level
simplify request handling
test integration reduce flakiness of error message test
tweaked modal title to better fit without wrapping
make my code prettier
update initializeservice js
add hour and minute for the post permalink
remove undefined group
remove blank line in block fix padded blocks eslint error
ensure consistent result
disable tracking for test runs
ensure to not crash with logged error
cleanup legacy backend event listener if modern backend is detected
slight tweak to avoid setting an invalid numeric value in props editor
fix eventual init issue in postinstall script
fix eventual crash in pre uninstall
remove spaces from cognito pool name
add debug mode
add showloading hideloading
fix taro weapp 修复state上空对象变为有数据时 子组件不会去setdata
make console override detectable for rn yellowbox
companion client better auth state check
companion handle status code errors thrown in google drive
style fix lint
update integration test
fix magnet and thunder protocol handle
always set pma
set pma to true
chore apply feedback about using nullish replacenode param
eslint config add componentdidcatch to lifecycle for react sort comp
fix create symlink use octal notation for chmod values
fix integration limit concurrency of bootstrap hoist test
test windows pass fix env to all exec integration tests
test windows don t assert on platform dependent env var interpolation
fix showloading returnvalue
fix rn 添加 pxtransform 的响应式适配
add jest test case for handling underscore in vertex name
oops pushed over pmuens change to bbpromise
use slss io
test pack directory remove irrelevant assertion
fix broken test
fix remove it only
update webauth js fix type environement environment
update cliauth js fix type environement environment
update cliux js fix type environement environment
hotfix fix config for scrollbehavior test
avoid problems with multiple cursors and lines
remove accidental log statement
docs mongoose add example to mongoose constructor
updated fixtures hash for tests
use declared sns topic name
attempt to resolve linting fail in travis ci
add pma flag
add spine runtime setblendmode monkey patch to avoid batch flushing
add commands that require aws creds
add commands that require aws creds
corrected object check to include url property
fix lint error
corrected debug namespace
flow type cleanup
re enabled packages backend build to be production mode whoops
fix lint errors
add uni service api alias
merge branch master of github com automattic mongoose
style fix lint
physicseditorparser has now been exposed under the phaser physics matter namespace so you can call methods on it directly
allow linecounter test to be run from anywhere
fix debug message when subscribing for changes
reverted a webpack config change that broke browser extension styles
test reduce clone parallelism
fix bug in password reset
fix replacenode is now defined
re add try catch inside agent s getidfornode
handle email password login attempts for accounts with no password
test fix failing tests that due to async support in generator
companion rename uppy file companion
chore add test for prop changes on replacenode
companion set allowed http methods internally
soy mode add select and plural
chore always upgrade vue cli service first
add shorthand alias for set constant js set js
add getvector methods
await ending workerfarm in reverse handle test
try another way of raising the timeout
moved mailservice stub closer to it s usecase
add missing dosettings call in updatesetup
test fix tests
use ctrl for jumptomatching command similar to vscode resolves conflict with ctrl p navigation on mac
fix uni request
add uni invoke api
fix accidental change
copy to backers folder
use inputtype do detect undo redo commands
make sure the editor doesn t attempt to render after destroy
avoid assertions for symbol when not present co authored by godfrey chan godfreykfc gmail com
fix intensity type
fix intensity type
fix without diff issues hopefully
feat nvue globaldata
if it won t render but the batch is open flush it
merge branch master of https github com photonstorm phaser
update fullpage js
correct dobra code and symbol
fix taro 直接引用 shallow equal 路径进行编译 直接把代码编译到 tarojs taro 包 不设置 external
update index js
update index js
updated flow type for devtools tab id
test bump jest timeout since vue eslint config are now externalized
remove scale abs as deeper effects unknown
chore taro quickapp 调整快应用运行时打包
fix failing test in ie due to attribute sort order
fix redux 修复小程序 redux 没有取消注册 statelistener 的问题
remove enterprise and platform from log info
newlines instead of commas
we ll use the browserslistrc config in uppy root instead
update comment on supported syntax for npm strip
asterisk mode add comment syntax metadata
asterisk mode fix bogus patch
uppy core avoid excessive setstate
improve variable name
improve variable name
feat add option to disable meta generator tag
do not pass handled errors via unhandled channel
expose stack trace for non user errors
recognize isusererror variable
treat unhandled rejections as uncaught exceptions
recognize yamlexception as user error
fix set english as default
add page rendered timestamp to telemetry
merge branch master of github com automattic mongoose
use slice literal instaed of prototype
optimize creating array of childrens
fix children array after review
style fix lint
setting assets data before loading extra compositions
add pretokengeneration usermigration cognito triggers
add missing runtask
add namespace for ts
frame destroy will now null the frames reference to its parent texture gltexture and clear the data and customdata objects
build app plus nvue
check that channel epg exists in buffer
add check channel data
better sound manager type
add clearcache function
clear cache before parsing each playlist
remove unnecessary output to console
fix demo completion provider example
chore skip confirmifgitdirty in api mode for now
test fix visibility issues in cypress test
update index js remove unused path module
expose errors during suite abort phase
improve path output
expose changed cwd error only on success otherwise it may be purely an effect of terdown not been propagated successfully
update index js add verbose mode renamed http to instance to avoid conflict in variable names
update index js turn off verbose mode
remove incorrect member state
update index js add debug argument to script
allowed access to post db backup endpoint no issue
fix www builds
eslint config base patch camelcase enable ignoredestructuring
eslint config base breaking func name matching enable considerpropertydescriptor option
eslint config base breaking enable prefer object spread
eslint config base breaking enable no async promise executor
eslint config base breaking enable no misleading character class
eslint config base breaking enable max classes per file
eslint config base breaking enable no useless catch
eslint config base breaking no self assign enable props option
eslint config breaking enable react jsx fragments
eslint config minor enable react jsx curly newline
eslint config minor enable react state in constructor
eslint config minor enable react static property placement
eslint config breaking enable react jsx props no spreading
eslint config breaking disable label has for enable control has associated label
update util js resolved cannot read property tolowercase of undefined error
fix taro components 修复circular为true却不会循环播放的问题
update format js add epg attribute to format script making epg parsing optional
fix compatibility with old node js versions that lack const
fix taro components 修复circular为true却不会循环播放的问题 fix taro components 修复circular为true却不会循环播放的问题
fix 去除 app 端 request 请求 url 首尾空格
update index js moved skipplaylist function to util js remove extra character
update util js add skipplaylist function
update format js add country filter to format script
fix taro quickapp 变量使用错误
assert ref is set
allow no scope hoist as a cli option for builds
remove unused import that s why you wait for the lint job to finish andrew
docs api fix typo mongoose prototype connection description
add comment about children validation
introduce wait util
move stage configuration to directory
introduce dx friendly function name validation
mark packages as public
don t default scopehoist option to true
minimize function name
configure tests for new naming functions
eslint config patch react state in constructor fix incorrect configuration
merge branch master of github com automattic mongoose
fix add property type for compatible with old version
clear invalid options
significant speed optimisation when large amount of lines are folded
website remove double escaping of inline code blocks
npm run prettify
add checkbox to next token in loose task list
fix for failing yarn dev on standard syntax check for mermaidapi js
bug fix handling non click graphs
update loadparcelconfig js
update react devtools inline to embed react debug tools since it s not published yet
update handleimmutable js check options not undefined
update util js add support uncompressed xml files
disable babel plugin transform block scoping throwifclosurerequired option for tests
standardised require call to use relative path no issue
update mp qq api
update mp baidu add getaccountinfosync
fix utils shallow equal 先判断是否为对象 修复 shallowequal true undefined 返回 true 的问题
configure google analytics
fix 修正错误的 js 类型
changed version name string to show version and date moved commit number into description string
update mp alipay api
temporary fix for scss import
update mp alipay api
clean up css fix
fix 修复createselectorquery exec 不写回调函数时在支付宝小程序报错
fix for test
update util js add list of supported epg file types
update mp alipay api
add code example for emberarray addarrayobserver
update explanation and code example for emberarray filterby
update documentation for emberarray findby
improved editing props demo experience for todo list
remove requestgraph addsubrequest
return null from logstat promise to remove bluebird warnings
move polyfill for codepointat to string prototype this method belongs on the prototype not the string object
caps lock bug template request for repro steps
update mp qq api
update mp alipay api
add test case for the hour and minute of the post permalink
revert a change in tag pages would cause a bug with tag page listing
simplify fix even further
remove useless check for null object with spread operator
doc add examples for emberarray invoke
changed default valid status config
fix a minor bug with snippet pages not displaying explanation
fix taro 缺少 hooks 函数导出
support fr units
remove errant describe only in unit tests
test increase waiting time before checking hot updates as the performance of appveyor build environment is quite poor
fix cannot read property log of undefined
extract long expression into local variable
doc add example for emberarray seteach
updated logretentionindays checks
chore stat lint
refactor to async await
improve var name
introduce wait util
fix test reliability
build uni app plus nvue
fix taro 修复小程序下 usepagescoll usereachbottom 等 hook 中 state props 不更新的问题
fix for issue with bindfunctions being called when there are none
improved github issue repro instructions prompt
test coverage for loading variable getters
wait to patch console until reporterrunner is constructed
hardcode cwd to parcel repo root for parcel babel register
feat taro 增加设备方向 api
soy mode fix setting kind on invalid tags
clike support nested comments in kotlin mode
chore opt add comments
fix raf if scriptlet bad proxy target it was working nonetheless which made me miss the mistake
doc updated emberarray invoke example
doc add example for emberarray property
do not log warnings on empty arrays
use absolute path in parcel babel register execarg
update test js
remove unnecessary mkdirp
fix inner audio fix typo
fix protocol request header contenttype
build uni app plus nvue
add google site verification code
merge branch cdxxii master into fix inner audio
improve multiline message output
ensure to use serverlesserror for user error
allow unresolved rest api id for tags
improve var name
fix serverlesserror definition so message in all cases turns as string
style fix lint
use shorthand method signature for afterprint in web app js
update uni vibrate
add a null check around memoizedprops
update destroy blueprint action so that afterdestroy lifecycle callbacks will still run by default
add support for if false in list directive processor to more easily comment out blocks of filters
prevent remove textarea form submit handler with opts leavesubmitmethodalone
fix cli remove check update error log
feat cli add todo
test error message
style fix lint
fix flushing problem with tests
fix console patching test by resetting modules
fix stat 删除统计日志
docs fix history version link
update util js
merge branch dev of https github com dcloudio uni app into dev
fix cli cssnano options
merge branch dev of https github com dcloudio uni app into alpha
remove unused custom var getter flag
disabled devtools tests from yarn build target
fix clear excessdomchildren for the textnode case
suppress act renderer warning for devtools tests
array reduce instead of promise reduce
add tag for canonical urls
fix a styleeditor variable resolution regression
allow passing non jquery objects as dropdownparent
feat cli enable mode for webpack in build
gherkin mode remove annoying console log
variable prefix is always a string
validate that options is an object if not a function
fix cli split chunks windows path
remove an unnecessary unused dt flow type
fix overriding package settings after packaging function
feat mp getmenubuttonboundingclientrect
rely on object assign
check for falsey variable resolver
remove dead code
fix variable prefix regex
the displaywidth and height should never be negative
在rn中的tabbar中 app js中配置的第一个page应是默认打开的tab
add newline between methods
doc update example for emberarray sortby
fix prettify issue
update packages ember internals runtime lib mixins array js co authored by ricardo mendes rokusu gmail com
fix app plus nvue targetstack
feat uni getsysteminfo 增加 brand 属性支持 app plus
feat helper hash url should have config url prefixed
remove failed golf attempts
update webpack config js
style fix lint
fix cli vue cli plugin version
easier to understand error message
less lodash more eslint
remove unnecessary return
fix sortobject should not introduce formerly non existent fields
style fix lint
remove enterprise from upgrade notes
fix for bug which doesn t properly honor defaults for justone
fixup test use page waitforfunction for hmr testing
feat cli wraps text in a text tag nvue
fix cli version
fix cli nvuepages
resolve empty config into empty object
clike fix kotlin indent for first line in lambda with params
refactor rename to bindhandles and execute before handling layout effects
docs add example for typeof a to distinguish typeof a from typeof a since they return different values
fix stat 修改选项卡页面获取不到标题的 bug
add comment to cloudfront distribution
rely on plugin paths not constructors so it works with refreshed module exports
ensure to work with tracking disabled
improve prompt names to make testing easier
cleanup test configuration
fix restore of process stdin
ensure working mock on windows
feat app plus nvue getcurrentpages returnall
fix module not found error detection
cover only plugin load with error handler
skip in case of plugin command only on not found error
chore remove extraneous log
test remove duplicated test case
add exit code
ci auto detect chrome version
ci log patched chromedriver version
replaced sortbytitle with sortbytitleandurl
moved skipplaylist to test index js
update index js add skipplaylist function add exclude option to script
in csrf hook only lowercase the route path if the route isn t a regex
moved skipplaylist function back
moved skipplaylist back to util js
refactor empty string constant
docs rephrase the upgrade notice
bump test timeout for slow test
update index js
fix code style
rendertexture would not update texture size on resize
update vertexcomponentcount in webglpipeline
style fix lint
fix rerender on ref equal context
moved injection logic to content script
mode meta add wl and wls file extensions
adds doc for map and mapby methods of emberarray
add the check for children
remove the scope and use the closure
fix cli performance hints false
remove done add some comments explaining the change
moved inline comment
fix lint rule
fix cli update browserslist
feat app plus add warning
updated for lint
simplified controller checks in get helper
merge branch master of github com automattic mongoose
doc release mark owner setuprouter as public
core change default toucheventstarget container wrapper
fix log usage for columnmigration command
rely on temp dir as resolved in serverless test
doc add example for emberarray hasarrayobservers
doc add example for emberarray firstobject
refactor gravatar utilize hexo util module
refactor utilize hexo util module
fix stat 修改统计上报域名
chore cli add warning msg usingcomponents false
test model fix tests
do async plugin init in load all plugins
simplified urlutils require path no issue this was previously going to a parent directory which was shared by both modules
remove the console from context test
address pr review comments
disable delayed drag events during destory
fix example typo for schema prototype plugin
fix needless visual toggling of update apply buttons
remove version line escape
fix lint update lint
prettify components versions
fix bug that continues to package optimize even when cache entry is found
build restrict root jest cmd to packages
build remove website babel config from repo root config
merge branch master of https github com nervjs taro
fix components swiper 从尾到头指定下标时的表现形式问题
remove the isconsumer flag since its not needed
fix row calc
don t use eval it breaks strict mode
a small optimization when updating documents
fix global tt
fix mp toutiao add eslint disable no undef tt
chore cli npm run info
build uni app plus nvue
companion logout revoke support for onedrive
use webext menus not browser contextmenus oops
update docs to match sort types
fix duplicate file test failing because doing new file sampleimage type image jpeg twice resulted in a different lastmodified
merge branch master of github com automattic mongoose
fix catastrophic backtracking in markdown hr regex
set dirty flag only when render state of bob is changed
update display origin when resizing
also resize hit area when resizing
tidy up comment for loadactionmodules
remove extra checks
test fix tests
feat 安卓端 uni request 增加参数 sslverify
fix app plus set default value for kernel to uiwebview usingcomponents false
build uni mp toutiao
revert to http
companion fix wss breaking change
improve consistency of return type
improve message formatting
expose setupaws configuration to enable customization in dashboard plugin
check for predefined credentials upfront
update typescript plugin patch version to match npm version
companion fix shared drive icon
fixes the previous commit
skipped scheduler tests using old auth method no issue to be updated
improve message for windows users
fix install command prefix is not documented and works differently on windows
workflow run checklink sequentially
workflow fix the maxlistenersexceededwarning
workflow try defaultmaxlisteners
bugfix release fix incorrect error message for octane features the correct command is ember feature enable sob weary
isassetinancestorbundles consider bundlegroup nodes when tracing ancestries
style remove unnecessary argument initialize
test remove console log
style remove newline
fix config is not parsed sometimes
increase default timeout
improve error logging
fix linting issues
append newline to avoid unnecessary back and forth changing
provider views display logout hint for a while longer
debug companion add other fields for testing
companion allow all fields for gdrive s drives endpoint
add isbrowser to resolve domexception undefined issue in node
check asset membership before remove from graph
remove includeall from traversebundle
safely remove connections between bundle and bundle group
update runtimes to reflect removals aren t references anymore
get rid of unneeded eslint ignore
ensured webhook handler uses members servicer getter no issue
wired up the members webhook handler endpoint no issue
passed appinfo to members api stripe instance no issue
add support for setting name in members api no issue
fix router 修复hash模式下tabbar不展示的问题
chore rollup fix require context windows
remove commented out code
refactor simplify project wide babel config
docs emotion ssr link
style fix lint
workaround for broken ctrl on new ipad
fix 修正 app端 settabbarstyle borderstyle 颜色
update nvue uni createselectorquery 移除回调中属性id的
fix 解决不带 tabbar 启动白屏的问题
fix force lines to be ended with lf fix windows tests
remove commented code
fix test timeout issue
merge branch fix test timeout issue
fix 修复含有 subpackages 时首页错误的问题
use correct check for nav button as target
better check for short swipes over nav buttons
multifile key fix
renamed member subscribed to member paid no issue to match the content gating terminology
moved author block and fix body reset issue
gameobject tojson will no longer output the scalemode in the json because it s not a valid game object property
texturesource setfilter will now set the scalemode to the given filter
canvasinterpolation has updated the order of the css properties so that crisp edges comes after the browser prefix versions
the buildgameobject function will no longer set scalemode because it s not a valid game object property
remove un used import
settransform will now set the imagesmoothingenabled context value based on the renderer and texture source scale mode
tilemap canvas render functions now respect game pixel art settings
set image smoothing based on renderer or scale mode
set image smoothing based on renderer or scale mode
style fix lint
add a code example to documentation for emberarray rejectby
add documentation to the reduce method of emberarray
fix flow types for creatable and stop spreading things that don t need to be
log sorted values on change
remove unnecessary console log from the example wrapper
improving sortable example behaviour
fix first child warning in chrome when viewing the site
improving async props documentation
passed basic updatemember function to members api no issue this adds support for updating member names
add the new customboundsrectangle config properties
chore undo deletion of space
chore remove spaces note to self don t use github editor for js changes
expose basecamera module
fix update ka locale weekdays
add video cache
expose the videofile file type
pass in the key and url
fix two cities stuck together in ru брянск and магнитогорск are different cities
fix absurd street name in ru речень улиц seems like a parsing error it means list of streets in russian
remove upward rootmode from website config as it is not handled through central config
don t render if there s no texture
update webaudiosoundmanager js
adjusted integration test
eslint config patch add static variables to sort comp rule
set part angle in body update into matter js
allowed overriding the default from address no issue this will allow the members service to pass a custom from address
changed imported modules assigned to const no issue updates to current standard
minor comment fix
refactor utilize encodeurl of hexo util
update error message
style fix lint in schema add
exposed member firstname in the theme data no issue this is very basic split on whitespace for now
add logging under cli for openwhisk nodejs
formatted log message
updated get set metadata fn signatures no issue this is to reflect an upstream change in members api
updated getmembermetadata to use findall method no issue this means we go via our version of the bookshelf model
switch to new date
texturemanager get can now accept either a string based key or a texture instance as its parameter
expose uppy on window for development as before
expose output in case of test crash
no url in constructor
feat mongoose allow setting schema option typepojotomixed mongoosewide
remove obsolete lint instruction
fix regression tests
fix remove unneeded validation check
test update cover test
wired up the note property to members api no issue
add note column to members table no issue
add gendered first names to de locale
add gendered first names to de locale
fix regression tests
remove text transform capitalize from buttons no issue button text should be sentence case not title case
add getsubject function for members emails no issue
ensured that members emails include our text version no issue
renamed getrequirepaymentsetting no issue the negation before this function call was a little easy to miss for me
fix regression test for canary content api no issue we now return the visibility field on the content api
fix viewport overflow in ipados safari
emit event after playback
add optional data object to stop method
chore do not log file change as error
chore hard set instances if not specified
add getvalue method back and updated it
update copy to examples js
your guess is as good as mine o
update updatelist js
remove commented out code
update build script
disable delayed drag events during destory disable delayed drag events during destory
fix 禁用多进程编译 避免manifest编译失败 fix 禁用多进程编译 避免manifest编译失败
style fix lint
minor refactor to validationerror
add member add and member deleted webhooks no issue
remove unused members getter no issue this would have errored when used
add add and deleted events to member model no issue these can be used to trigger webhooks
fix minor bugs spotted during code review
fix correct order for declaration
provider views object spread co authored by renée kooi renee kooi me
provider views remove debug log
fix hide forked process window
fix options explicit pre dist tag
fix options explicit force publish
fix options explicit conventional prerelease
fix options explicit conventional graduate
fix options explicit use workspaces
fix options explicit force local
fix up jsdocs
aws add deployment bucket policy to naming js
chore forgot to run prettier
check to see if the environmet has video element
fix providers list help output in config command
document reasoning on error silencing
clear eslint workaround
expose valid providers on serverless commands
improve internal representation of validproviders
resolve bug restricting translateto to specific bounds
fix type import
simplify options resolution
seclude tabcompletion utils folder
remove eslint disable comment
remove obsolete comment
chore add links to sort and splice tutorials
feat package graph deprecate method prunecyclenodes
minor update to argument names to better adhere with existing translate documentation and code
disable gamepad api workaround
fix a bug in case if the environment does not support htmlvideoelement
small gramatical mistake
updated fixture integrity hash no issue
run sanity test in non supported environments
ensure test runs in isolated environment
add spacing in error message
add spacing to constructed error message
update ontouchmove js
fix no need to assertcliversion avoid breaking old versions
fix rules fix typescript rules
fix typescript rule semi
add centeredslidebounds parameter prevents gaps at beginning and end of slider
promisequeue make queue length public
workflow keep all packages versions in sync to reduce cognitive load
workflow correctly set releasetype
fix test for updated favicon no issue
feat rn 添加 tabbar borderstyle 支持
shouldn t remove the blob data or videos too large for memory cannot replay
iterate nodes to copy instead of traversing
undo brackets removal
fix cli babel loader
fix warnings of null currentnwwindow when ondocumentstartend is handled
fix cli remove entry app
fix arn name
fix undefined issue
update build js
add mobileviewdistance config key
change md file for parsing
remove flawless feedback banners
update connection close docs
change link in plugin button in header
add meta image for designplugins
chore lint error
test geojson tdd case markersinheritoptions see next commit for resolution
fix 修正 tabbar borderstyle 颜色
fix 配置 tabbar selectedcolor 默认颜色
remove redundant condition
fix mp shadow css
exclude rtl direction mark from autocompletion word regex
fix 调整 app 端 tabbar borderstyle 颜色
modified checkoverflow to account for slide offsets
update post process to recive final assets rather than initial ones
refactor arrow function
fix issue with wrong row column if not full groups
fix mp add reflect
chore sort changelog tags by date
ensure to remove the stack if some tests are aborted on the way
fix eslint configuration for test utils
ensure to rely on preprepared env
prevent noisy logs output
improve timeout setup
remove obsolete comment
fix failing tests
add mscssmatrix to matrixfn options
refactor test assert on loadjsonfile registry keys not snapshot
chore create release ignore default case coverage
add unhandled promise rejection handling in suspense tests
skip unmounting test for now
updated variables test to include tobool casting
bug fix in tobool casting
bug fix in tobool test
fix duration when freemodesticky is active
ensure alphabetical order
update code to return promise
refactor destructure util module
fix enforce process send to sysinfos
fix less aggressive timeout
ensure boolean result
fix tabcompletion issupported test
chore increase mem warn color
using strtobool in tests
reorganize custom role setup
fix version correct warning message
update emberarray invoke docs to use native class
fix dependson handling
merge branch master into reject doc
add missing parameters for addgltexture method for ts def
fix wrong function name of ts def the correct names are in index js
fix custom resource tests
fix eventvalue for categories
add abstract banner on visual debugging tools section
markdown boolean data attributes will now be add to elements too
fix exception when sending email of messages attachments undefined
add comment explaining fallback check
add example for emberarray isevery
ignore defer definition in coverage
small improvement to hashstampedtoken
remove naming function for logical id
fix visibility of abstract plugins
make link tagets blank
update variables test js
remove no longer necessary noyieldsallowed wrapper from loadisopackage
add posterize time support
fix return value in reject code example
add code example for emberarray isany
chore performance update
feat taro 新增 getselectedtextrange 和 hidehomebutton api
fix lint issue
update array js
update variables js
update variables js
remove debugging output
add error handling to loadepg
update variables test js
skip an untested line
use cross browser helper to create event
add centeredslidesbounds defaults
update array js
update array js
test fix typo
add fallback timeout
merge branch master into feat improve coverage
use getdisplayname in a couple more places in debug
skip compat string ref test with explanation
converted has helper imports to individual const no issue
chore cli perf msg
update variables js
fix typo for namespace
update packages ember application lib application js co authored by ricardo mendes rokusu gmail com
docs model remove unnecessary
spelling in case
spelling yoyo ing
fix taro quickapp 修复快应用事件绑定异常问题
feat audio context
update index js
add tencent provide in create test js
add runtime in create js
test pass with underscore in a class name
simplify reporter unit test setup
fix linting warnings
fix failing test due to prettier formatting
add braces around multiline if statements
update variables js
handles the error thrown case
reformat karma config
fix lint errors
fix broadcast componentname
chore reduce col size for ns
update index js
updated jest test cases for pie diagram
fix the set hook ref
remove obsolete lint instruction
ensure default bucket policy is not configured for custom bucket
update variables js
apply suggestions from code review co authored by ricardo mendes rokusu gmail com
feature thumbs add option to control whether multiple thumbnail slides may get activated or not
change the strange undefined assertions into more meaningful ones
update variables js
add support for the shadowdom
fix line length issue
fix oom crash when cdata tag unclosed
feat cli usingcomponents true
renamed getgroup to filtergroup
update update readme js
update update readme js
fix devserver flow violations with mime
fix og locale ignore case flag is not necessary
fix up jsdoc comments
fix options devtools test
add comment for another test that won t work against minified builds
update generate js
fix node test imports
fix getdomsibling test imports
point isvalidelement to same file browser tests are running against
add flag for sending email when post is published no issue
swapped handler apply after if in class js
improved bulk email service no issue used proper logging
fix bulk email service loop no issue
fix db integrity test
fix custom subjects for mega emails
improve location of tencent templates
fix error in calculation of the number of channels
introduced new error message for position fix and scroll
update code to not use regex lookbehind
add code comment
chore cli perf msg
skip processing on race condition
ensure to not crash on race condition
fix nodejs core module in web browser ignore url module
omit s r properties in fromt matter
fix lint errors
add together mode
invalidate asset requests when included files change
fix post serialisation
add email preview endpoint to admin api
fix linter error
add content gating to the mega service no issue
fix jsdoc comments for bulk emailer no issue
add test for multiple class diagram
fix app plus nvue getappwebview
fix nvue native event bus
filtered out members which are unsubscribed in mailing service
remove console statement
remove unwanted karma file
simplify istanbul config by specifying include array
fix comments typo
allowed status data for email preview api no issue
add date formatting for email post data no issue
fix permission for test email api
fix email preview auth for test mails no issue
fix input event bug in nested container
fix members settings cleanup for mailgun config
fix tests for pages api no issue
add eslint ignore comments for template no issue
add defaults to member model no issue
allowed getting member by uuid no issue
add uuid column to members table no issue this will be used for unsubscribe links
refined email template
fix db export test
fix typo in email permissions migration
add field defaults
fix db version integrity check
fix api onpush
fix db integrity check
add current year in email footer
add unsubscribe url
feat mp weixin add options
feat mp weixin add options
fix test preview mail for draft posts
updated default values for post model no issue
fix cli vueoptions
fix cli pluginoptions
passed unsubscribeurl template variable to bulk email no issue
add status all by default to preview api
refined figcaption size in emails
fix unsubscribe url for preview email
make checking for error messages more robust
add test for preview email subject
fix options not being passed
save renderer to inlinelexer options
remove bench debugging
add do not edit comment in compiled files
clean up code
remove unwanted comments
serialized post html for email no issue
add comment to addemail method
fix typo on email relation in model
not creating email record when there is nobody to send it to
changed existing email status handling according to chenges in schema
fix pre style for email template
fix email not returned in put object on post publish
fix gallery image height
fix clicking select tag
chore cli warn stringify
chore cli add log
fix generic use breaking babel parser
fix rollup s stream warning by moving require down
fix findlaststyletag for text nodes
fix groupedtag buffer resizing oops
fix rollup plugin babel warning
remove stylis rule splitter from stringifyrules
filter out empty component styles from ssr
fix linting error
remove an allocation
shave off some unnecessary bytes
fix cgs determinetheme usage
revise simplify theme determination
fix the binding function error
sighs in javascript
treat falsey deploymentprefix as empty prefix
fix linting error
set from adress in the mega service no issue
updated name of bulk email service config no issue
fix typo in regression test
fix targetgroups tests
feat auto enable sysmonit when command typed
fix make status col larger for long status
thumbnail generator emitpreprocesscompleteforall
fix vue add should not crash when executed under yarn pnp todo should fix loadmodule in pnp environment
add reset for testing purposes
rename parse argument to tokens
remove unused function
mode meta make findmodebyextension case insensitive
docs codeblock document each parameter
feat codeblock add wrap option
test codeblock wrap option
docs codeblock use options to denote it s optional
workflow don t skip version prompt for local release ci skip
workflow should be able to publish to next dist tag in local registry
test aggregate fix tests
update csound urls
create nojekill file
fix tests remember to return promises
make code smaller
add selectprops documentation and how to use it for passing dynamic custom style
typos and bad translations redone
typos and translation redone
companion log error if exists during token verification
remove nojekill file
fix weex boxshadow
merge branch dev into alpha
update copyright year co authored by steven steven ceriously com
update copyright year
update packages ember application lib application js co authored by chris garrett me pzuraq com
remove unused chars
add nojekyll file
only allow explicit deploymentprefix for empty
tolerate variable startup message order in static html self test
reverting karma conf
code golf suspense js
remove unwanted conditions
updated by alf automation
ensure to treat filestochmodplusx as an optional
create nojekyll file
allow null undefined
remove now unnecessary code
don t expect certain values for inlinetype
fix connection bubble up connected disconnected events with unified topology
fix bulk settings input serializer
don t extract pipelines from entry specifiers
fix post not showing authors in email no issue
fix mailgun instance not updated on settings change no issue
fix settings tests
updated fixtures regression test
add site title as sender name for bulk email no issue
fix typo in banner
update cta link in banner
allow test to run
normalize empty class attribute in test assertion
fix 解决uni chooseimage在ios企业微信无法使用的问题
fix gatsby transformer documentationjs add missing quotes around field names
fix lazy ref test to not rely on internals
remove src import from test helpers
remove test entries
debug log oauth response when access token is absent
chore fix test name for queryparams let test
remove duplicate omitundefined options
fix do not throw error on dump not present
merge branch master of github com automattic mongoose
style fix lint
test re enable wrap
fix overwriting model name
updated test email to throw error for failed requests no issue
docs codeblock update description of parameter
docs codeblock wrap option
docs codeblock fix line order
docs codeblock update mark description
fix default disable highlight hljs by default
fix test failure in prod
update page parser js
update page parser js
chore add helper message that cron is still online even after stop
remove unused import
remove superfluous typeof
use default db if provided in connection string
add test for default db in uri
fix connection only buffer for open events when calling connection helper while connecting
correct version number
add mipmapfilter to game config
corrected typo in test
fix conventional commits ensure potential validationerror in getchangelogconfig is propagated correctly
test version pass explicit no ci to ensure error is thrown
update integration package tests with new runtine
rely on new runtime in custom resource lambdas
fix cli partialidentifier cache loader
test empty deployment prefix
fix cli pitcher path
trim trailing spaces from string values in advanced settings
fixes rollup freaking out over an eval
updated by alf automation
error message improvement
simplify the return expression as this is a boolean
fix 回滚误删的event bus
fix template path
updated by alf automation
add separate render test
fix typo and linting fixes
run prettier on updated files
style fix lint
test fix tests
add gettype test case
fix app plus getcurrentsubnvue
fix app plus restoreglobal
fix nvue native api
feat core add beforesuccess
show the time in seconds rather than milliseconds
feat core set default value for extras to
update bridge js
fix core get clipboard data set clipboard data
fix core keep extras alive
fix 修复 rc shared scroll js 与 src shared index js 循环引用的bug
simplify check for truthy value in allowcredentials
don t reset text
don t throw error on job rejection
feat cli vueconfig function
feat cli change svg media limit
fix cli svg loader
chore fix typo in comment ci skip
enhancement remove console log in dist files
update cloudformation configuration
override default curve getutotmapping method since u is equal to t in linecurve line getutotmapping could return u directly
remove il variable
updated by alf automation
merge branch master of https github com photonstorm phaser
ensure spacing on if expressions
support semanticolor syntax highlighting
fix remove context status from websocket access log format
fix old style tests
mark arguments as optional
fix bug of game destroy
chore cli add block test
fix post serialization to keep plaintext in json no issue fixes missing plaintext on email preview fixes tests
fix url import for post email serializer no issue
fix test an dcomment
different combine threshold
fix members test no issue
workflow correctly decide which dist tag to use
custom render values set
companion onedrive domain validation
style fix lint
disasble csp test for now
fix linter stray whitespace was commited
change param type from string to int
fix request content type application x www form urlencoded 数据格式不正确问题
fix tabbar indexof
update matterworldconfig js
update manifest 支持配置屏幕启动方向 无需打包后生效
chore run prettier
breaking test into two
remove no longer used property
typescript improve pick shuffle types
unnecessary flattening and interleave of css without interpolations
improve configuration extemption
style fix lint
test fix tests
fix tests setup
do not update external apigw configuration
add missing plugin configs
update config for new tooling
style eslint arrow parens with as needed
fix test name
update test after recent changes
fix resolution of custom rest api id
update mattercollisionevents js
update collision js
don t count slides with class uncounted
skip matching a body against itself
updated custom render properties
remove old properties and add new ones
updated testcase for distance calculation
add some comments
fix aws creds handling
chore connection set driverinfo option
fix col size when tracing enabled
chore undo change to spawn code
load images from atlas path
creatable add option meta field to onchange for create option action
companion set required content type for onedrive domain verification
companion send content length for onedrive domain verification
companion use onedrive validation content verbatim
fix suspense component detection
companion restore compressed content for onedrive domain verification
test expose cachedcredentials mishandling in invokelocal
fix update notification locale issue update notification was shown in english for all locales
chore typo in comment
minor code review
fix changing global defaults
being clearer that we are using function bind with global scope internal cleanup
chore remove debug code thats causing tests to break
add additional ssg transform test
rename uuid to nextid
fix ensure to not create cognito pools marked as existing
remove unnecessary check
update documentation for shouldupdatescroll co written by the wonderful madalynrose
re factor getmainthreadworkermessagehandler to support arbitrary global scopes rather than only window
fix navigateto callbackid
update gettype test js
refactor cleanup conditional
refactor abort unnecessary work
temp bounds debug
fix 解决纯 vue 项目配置 condition 报错的问题
nvue ad 编译为 u ad
new frompath reg exp
merged new pointaworld and pointbworld functions
remove use of common indexof no need for it
update cli to work with latest commander js
gantt callback default to node id as callback arg
fix box always ignore git folder of themes
code style fix
remove extra space
another style fix
adds simple tests for parsing callbacks wrt args
refactor set default pattern
update physicseditorparser js
test ensure reliability of custom resources generation when testing
test fix fork test i accidently dropped the part of the test that checks that the script correctly started
log error stack on exception in publication
remove deprecated code
renumber render tests
fix loaded event in delayed callback
update oauth js
update screenorientation 兼容微信小程序 pages globalstyle 配置
remove worldconstraint changes
still fix sleeping
bump version of ejson package
add test case throw the friendly error for ejson stringify function
fix 修复connection effectivetype可能为undefined情况代码报错
fix pages json 没有 globalstyle 节点时编译报错的问题
refactor renderer arrow function
print warning for missing babel pluginsco authored by sven sventschui gmail com
add note about hoisting
disable sync verts for now
refactor theme sub classing
test remove extra async keyword
build service runtime esm js
fix release script to ignore empty package folders
test unmock collect updates when testing publish from git
fix typo in log
fix prevent uncaught exception in case of open util issue
feat cli add filename for compileroptions
fix tests no issue
add jsdocs and property
per body collision callbacks
update physicseditorparser js
update protocols js
fix fix locale handel unsupported locale
fix ensure not to autocomplete hidden commands
refactor cleanup variables
test increase timeout
test improve title
fix semicolon in updatelist doc
style assign to this variable
add doc for gameobjectfactory register and gameobjectfactory remove
fix jsdocs name to method
add optional params to parsefromatlas doc
refactor binary installer ensure latest stable npm version in bundle
chore register binary installer scope
increase maxbuffer to fix enobufs error
remove extra elements in navbar
fix self test error unknown package in top level dependencies jquery
feat cli add usingautoimportcomponents
fix getapp 失败增加提示
update protocols js
chore register variables scope
fix nvue remove exclude
violet theme implementation
small optimization to modified defaultresumeloginhandler
fix mp multi event in v for
test alphabetical order on exports
chore register plugins commit message scope
fix git graph not clearing
fix lint error
update protocols js
refactor cli usingautoimportcomponent easycom
chore register user config commit message scope
remove useless console log statement
soy mode add support for default param value
chore register aws lambda commit message scope
refactor aws apigw convert trailing arguments into options
chore register aws apigw commit message scope
refactor aws lambda improve var name
fix error with filter element
add particleemitter settint
throw error for a bad audio asset key
add particleemitter remove and particleemittermanager removeemitter
allow cache keys to be shared between archs
bracket notation for type phaser tilemaps layerdata data
style router clarify function name
other typos in flowdb
docs add example to schemanumberoptions enum
fix mousewheel fail
fix cli condition
add test asserting the callback to be called
fix lint error
crop component is missing from rendertexture class
update 回退将nvue ad 编译为 u ad
fix mp alipay add createselectorquery createintersectionobserver
fix mp alipay add createselectorquery createintersectionobserver
playground fix codelens provider example
refactor cli reuse same argv slice will allow reliable memoization of argv resolution
chore register cli commit message scope
changed tile gettint function to use gettintappendfloatalphaandswap
refactor test extend tag destructure
fix indentation after a literal literal block
remove render opacity as it s not valid for constraints
update bridge js
fix code style issues
improve typings generateframenames
style fix lint
use r instead of duplicate ref
docs model clean up some docs re aggregate
merge branch dev into alpha
fix standalone ensure dashboard plugin policies are bundled
fix standalone ensure dashboard wrapper is bundled
chore improve commit message scope name
style fix lint
docs for phaser sound basesoundmanager play always adds a new sound
feat mp weixin getsysteminfo safeareainsets
build mp weixin
remove redundant method call
minor code review
feat cli improve log debug mode
xhr upload do not emit limit warning if an existing rate limit queue was passed
locales lint fix
style fix lint
elm mode remove tab character
test improve util signature
test support checkiscomplete option in confirmcloudwatchlogs
merge branch dev into alpha
update matterdebugconfig js
remove old comments
final config structure and values
you can now set gravity false in your matter config and it will reset gravity from the defaults to zero
reduced default anchor size
fix comment about the published data structure
tweendata now stores its previous ease value
emit the current and previous values
build app view
inversemass is a required property also remove duplicate jsdoc
start of alignbody method
adds galaxy hosting url when creating new meteor apps
fix twice jump to message
test fix race condition
add bodybounds class
fix build remove reference to replace array includes with indexof
refactor aws lambda do not deep merge when not necessary
refactor aws lambda resolve deep value once
add fromphysicseditor method
check for disabled body in collidespritevsgroup
add fallback to empty object for default options
remove un used code
more robust checks
remove un used properties
merge branch dev into alpha
update factory js
fix set new custom fields values for livechat
fix bug in physicseditorparser where it would allow non clockwise vertices through
setbody can now use json data natively
handle scaled bodies properly
add conditional to fallback to old style
address code style issues
fix css 标签排序后在头条小程序显示不正常的问题
fix usual breakpoints
signal and severity are vhdl keywords not types
update polygon js
add autofocus to directory
fix lint issue
merge branch master of github com automattic mongoose
fix request api 增加详细错误信息
use fuzzyequals for safety
remove namespace as it conflicts with typescript defs
use cordova lib api to run cordova
beta versions for mobile packages
update matterphysics js
bump mobile package versions launch screen mobile experience and mobile status bar
add a converter on getting the file buffer from apps engine
don t wait for mongod to step down on shutdown
exposed staggerbuilder and add missing param to jsdocs
test fix race condition
style fix lint
fix error when touches is not set
merge branch fix tabbar into alpha publish
merge branch fix tabbar into dev
style fix lint
fix pathfollower always at end of path
add default settings to a new mongoose instance
remove set from schemaarray
use a fake option for testing
default options to an empty object
fix boundingclientrect bottom
test wrap error test with consola to make output clear
test fix uncompressed resource size
add settintfill method
always flush because tri strip
update fix spaceid error
chore remove unnecessary log statement
test avoid unexpected outside stderr assertion count
remove unnecessary uses of innerhtml
chore cli update tests
debug make tests pass
merge branch master of https github com photonstorm phaser
add label to formelements
fix incorrect xcode casing
update geteasefunction js
chore add cross env to get unowned packages
refactor convert statuscode to code error parameter
test fix socket hang up test release
code style fix
style fix lint
create tilesets from image collections
xhrloader will now use the xhrsettings withcredentials as set in the file or global loader config
config loaderwithcredentials is the new global setting for xhrsettings withcredentials
set depth of input debug body
skip custom element tests on ie
remove unnecessary constructor
spec text editor component linter test var fix
switch demo to new devtools bridge
dart mode add yield keyword
set nodeintegration to true
fix bug that was making us to reload the page after cancel was clicked on editing
correct entitlements path
add horizontal parameter
improved issingular function
push to the updatelist too
add predestroy handler
fix test issue
fix this return type for phaser gameobjects dynamicbitmaptext
style fix lint
test remove only
test skip the eslint migrator test that constantly times out for now
chore add newline in prompt message
test fix mock module in upgrader test
style remove unnecessary blank line from webpack config
fix centering body in setsize
merge branch master into node submodule exports
oops missed the only again
fix typo in error message
remove unnecessary merge of schema with itself
chore update chrome ua
add more tests for compat isvalidelement
update ropecanvasrenderer js
add alignment property
update rope js
update rope js
undef or null
skip map iteration in children
add the path to the parentstack to make it easier to see where it came from
add a check that options exists to fix unit tests
update mattertilebody js
update transformmatrix js add missing braces
fix this return types for phaser input keyboard key
merge branch jsdoc of https github com jasonhk phaser into jsdoc
fix standalone ensure to use proper cli params parser
use phaser display align enum instead of integer
fix this return types for phaser cameras controls fixedkeycontrol
fix this return types for phaser cameras controls smoothedkeycontrol
fix this return types for phaser cache basecache
fix this return types for phaser curves spline
fix this return types for phaser geom circle
fix this return types for phaser geom ellipse
fix this return types for phaser geom line
fix this return types for phaser geom point
fix this return types for phaser geom polygon
fix this return types for phaser geom rectangle
fix this return types for phaser geom triangle
fix this return types for phaser input mouse mousemanager
fix this return types for phaser input touch touchmanager
fix this return types for phaser plugins pluginmanager
use proper popup url
found little typo
feat add run mode config
fix get unowned packages
api expose getcomputedslidesize for plugin developers
chore update index js
chore update index js
chore update index js
test don t check for taobao registry during tests
fix correctly catch errors of offline install
remove preventdefault method this method cause editor to open virtual keyboard when scrolling and has focus at the same time
update serve js
correct position in constructor
correct center in constructor
correct dimensions from game object
fix wrong body dimensions from a scaled game object
update performance test import
enable code coverage
run isvalidelement tests against minified output
remove console log from test
change from exclude to include pattern for global config
add default settings to a new mongoose instance
use a fake option for testing
add jsdoc to schematype set
test uncomment the airbnb eslint migrator test
fix possible unintended scrolling in oninput test just like we fix it in hyphen and localizednumber tests
fix don t output warning message on eslint deps upgrade
fix jsdoc warnings
impact physics is now remove from the core api
add more comments about data mutation assumptions
bump minor package versions for livequery changes
formatting fix prettier to build script
merge remote tracking branch origin develop into release candidate
fix fix duplicate slash in metadata url
fix should use the local version number if the cache falls behind
fixup fix fix duplicate slash in metadata url
refactor use dep versions from plugin package json
chore register aws http api commit message scope
add enablecallback option for appcache
remove console log
soy mode remove templates property from state
remove redundant live region role
making getid more robust
update env js
fix driver from only running vim tests
fix clear require cache after upgrade before migrate
fix migrator correctly extract config fields to files
chore mark jsdom as external to remove warning
fix adapt the new api of electron s touchbar
companion fix tests
add new property synctorender
update uni createrewardedvideoad 修改错误回调字段名
fix ui fix the incorrect regexp used for cors check
eslint config base fix indentation with jsx fragments
fix github deprecation warning
refactor test useful rest parametors
update tileintersectsbody js
update tileintersectsbody js
update gettileat js
update gettileat js
update tile js
update tile js
update tilemap js
update tilemapcreator js
update culltiles js
update gettileswithinshape js
update statictilemaplayercanvasrenderer js
update statictilemaplayer js
update layerdata js
update parsejsontiled js
update statictilemaplayer js
update dynamictilemaplayerwebglrenderer js
update puttileat js
update tiletoworldx js
update statictilemaplayer js
fix line numbers
checkcollision none should prevent tile collisions
exit early for checkcollision none
refactor make tracker unique
fix system engine config keys
fix using wordpress example
update app端可以只传files 不传filepath和name
updated acceptance test to include db export request check
refactor utils fix eslint coma
refactor eachasync remove trailing space
refactor applyhooks fix poping last arg from args remove from args
bugfix lts add test ensuring singletons instantiated during destruction are destroyed
use gameobject s world position as the debug shape position
fix issue where test page does not run in ie
adjusted expected values in tree sitter failing tests to match grammar changes
fix lint error
update src rules js simplify code regex co authored by tony brix tony brix ninja
useuniqueid now returns the same reference every time
giving useuniqueid memo function a name
remove on heading regex unneeded on regex heading detection after tables and paragraphs
adjusted expected values in exception reporting specs
send inline html to renderer
add inline html test
fix blockquote end check
style fix lint
refactor taro 调整包的导出
style tests remove comment
update upload file type
remove needless semicolon
fix 某些场景下 request 总在超时后收到原生层回调
bugfix beta fix setdiff computed macro used within glimmer component
eslint config add config for disabled new rules
fix small typos in documentation
fix tests no issue
fix css 标签排序在头条小程序上导致样式不正确问题
fix members tests no issue
style fix lint
merge remote tracking branch origin develop into release candidate
add support for parser functions lua invocations
fix enter behavior
test more test fixes
fix typeerror for headless renderer
fix bug in test logs
add note about test
remove commented code
test aws http api fix typo
update app utils client lib restapiclient js co authored by diego sampaio chinello gmail com
feat add trackers blacklist constant
stex mode ensured that tag does not clash with object prototype properties
verilog support folding by indentation
add missing feature flag
ignore text vnodes
companion only validate url for fqdn when debug is off
refactor launcher callback
fix use of deprecated function in test suite no issue
more accurate usage description
merge branch master into master
updating comment typo
remove useless slash in path
refactored variable name to match intent no issue
add behaviour switches
coverflow stretch percentage for responsiveness
protobuf mode enable brace folding
fail graciously if no benchmark dataset found
handled missing resource path for resized images no issue
use correct element id fenix popup panel
remove setting asyncfn name to fix a safari bug
chore fix typo
feat mp add shoulddiffdata
chore use transloadit com in tests
update tag js
chore fix publish on windows
exclude a debug path from coverage
fix variable name
test improve test conditions for ip blocker
test fix truthy check
fix lint fix lint error require atomic updates
rm mediawiki tokens
bug fix spelling errors in the test file
update the xmlbuilder isopack s package js file
refactor aws http api improve config resolution
fix mp add error
trigger onmenuopen within handleinputchange only if the lenu is closed as suggested by pruik
disable hardened runtime
exclude another debug path from coverage
handle account verification mismatch
workflow allow manually specifying dist tag when releasing
fix cli compatible with cnpm
fix app端 uni request content type 为 application json 自动序列化对齐微信小程序
only transition properties that have changed
add mw mode to modelist
update component base parser js try add more support to mp weixin component options setting such as styleisolation
fix flash message in case of a rare database issue
修复 app json 被部分覆盖的问题
companion read state from session in oauth redirect controller
add comment to clarify string concatenation
companion use state in session in tests
add child entitlements
allow build to continue if notarization failed for testing pruposes only
docs radian values for transform rotation
add redirect support
fix standalone ensure to bundle local invocation non node js artifcats
chore do not restrict lines length in body
chore extend list of accepted scopes
chore ensure alphabetical order of scopes
chore extend possible message scopes
fix components 数组没有 split 方法
add isdefaultprevented fn to event
expand test coverage
增强微信小程序component构造器中options的配置 让其支持更多options配置 如styleisolation
更新 component base parser js 简化options配置代码
add const to for in loop
delimit redirects w whitespace
refactoring synctorender to fixedstep
merge branch master of git github com jaceechan uni app into dev
updated dynamic linking for spell check
ts def improvements
delete mangled variant
fix typeerror invalid calling object in ie
fix standalone chalk is an object not a method
fix condition for emmiting slidechange event
add nord to theme list
fix theme name
fix key name for nord theme
fix cli const var
merge branch dev into alpha
feat cli optimize hmr
add captureexception helper to sentry integration no issue this allows us to custom capture exceptions
captured bulk email errors in sentry no issue
exit early for checkcollision none on bodyb
fix aws cognito fix pool update handling
chore register aws cognito message scope
docs fix incorrect type for fontweight
ensure that renderqueue is always sorted by vnode depth
remove an unnecessary check
remove an unnecessary piece of code
refactored members csv export for bettere readability no issue
change image api base url to stable provider
fix 修复content type未忽略大小写的bug
use quiet option for ios to avoid unnecessary messages
replace underscore call
update themelist js
fix sharewithsystem type text
merge branch dev into alpha
add site url to update check body no issue
fix qa getprojectname
update example recognize js
control arrows fix visibility bug in linear navigation mode
test fix tests
test fix tests
fix 修复安卓调用 setsoftinputtemporary 后 adjustresize 配置失效的问题
fix qa package
chore qa remove todo
fix app nvue v slot
feat wxs support alias
replace hasownproperty with undefined check fixes test failures
fix undefined state during processing
don t expose translation api keys to client
fix wrong container getbounds with child container
test remove old fetch warning check a test to ensure legacy works tbd
code clean up
code clean up
fix createbleconnection 无法进入回调的bug
change variable name
use indexof instead of includes in test
application name correction
only announce current fragment to screen readers
addressed review comments
fix uni showactionsheet 参数 itemcolor 无效的bug
updated by alf automation
fix app uni setnavigationbarcolor 无法改变返回按钮的颜色的bug
add owner stack for invalid child warning
small for for merge issue
changed a few things preparing tiles
fix connectsocket protocols accept string
fix test time is omitted in front matter
test fix tests
fix app平台修复websocket onopen时发送数据报错的bug
style fix lint
style fix lint
reduce comments related to controller initialization
style fix lint
merge branch master of https github com adobe brackets into graphicsfileexternalapplication
read billing url from config and add to api no issue
fix regression test for database migrations no issue we add a new permission and this test was not updated
eslint config base patch allow triple slash comments
eslint config base patch disable prefer object spread for airbnb base legacy
chore register templates commit message scope
refactor aws http api ensure function name in a warning message
fix aws http api do not validate timeout when no httpapi event
tidy up for lint
fix qa output optimization
refactor cli 模板 autoprefixer 配置移动到 package json
webaudiosoundmanager description spacing
fix umd builds reactsharedinternals reactcurrentbatchconfig suspense does not exist in reactsharedinternals umd this pr adds it
vim bindings disable autocorrect for prompt
updated by alf automation
description for pack property
fix cli easycom
merge branch dev into alpha
on wiindows path is converted to windows style when opening with external editor
merge branch dev into alpha
updated strings for graphics file association
updated strings for graphics file association
fix png and gif support of extracting exif
addressed review comments
add optimized tests
add strict equality usememo test
test move scope of final cleanup
rename removetext to removefile
fix fs test js
used proxy in helper rather than direct require we already have access to urlutils through our helper proxy
add html to textrenderer
test fix tests
use golang style for catching error in test refactor
await configuration loading on windows
physics groups descriptions
tweak build script to not corrupt import fresh
fix message in release script
made a couple accidental line spacing changes prettier not sure
simplify tag regex
canvas renderer typo fix
remove redundant assignments already picked
add point property for points https doc mapeditor org en latest reference json map format point example
offset by object data dimensions allowing zero offset
fix cwd cli option description run target script as username run target script from path cwd
add phaser types tilemaps tiledobject point
chore ci 尝试更换 image 测试用例的图片
chore cli remove unused
force ci build
netwrk path also need to be converted to windows style
companion test remove flaky test
docs link to mongoose promise tutorial
dialog addon don t close dialogs when tab loses focus
addressed review comments
modify test description
fix step easing by implementing jump start method
description for basesoundmanager playaudiosprite
fix strict vnode equality
clamp volume fixes indexsizeerror the index is not in the allowed range
fix 修复app端uni canvasputimagedata无法正常使用的bug
add from cache message when loading traineddata
merge branch master of github com transloadit uppy
remove only from regression test
fix babel config in our files
adapt rollup config for recent babel present env version
switched sentry to use helper file for ghost version no issue
add missing semicolon no issue
graphics file assciate should get triggered after uuid is generated
fix cli remove exclude options of babel loader
adds name to store for devtools
add index to spine atlases when loading through asset pack
update number js
style fix lint
remove mongoose set debug true from test
docs fix my math in transform rotation
style fix lint
fix debug code breaking tests
merge branch dev of https github com dcloudio uni app into dev mp welink
feat core add getmenubuttonboundingclientrect
chores remove blank line
style fix few stylings issue
fixup self externalization
update 增加getprovider方法获取广告服务商 增加服务器验证回调事件
update 激励视频 onverify
move history api feature detection to utils
fix ios 端第三方输入法收起后点击屏幕会重新出现的bug
bump patch versions of webapp dynamic import and appcache
bump patch version of caching compiler package
update 修改激励视频服务器回调参数 增加 destroy 方法
quickapp 更新serve判断逻辑 windows平台 为
add usemaxwidth to staterenderer
chore pre release sync
update src diff children js co authored by cristian bote cristianbote users noreply github com
remove redundant cast
test fix create react app test
merge branch dev into alpha
remove stable branch warning from gruntfile no issue the stable branch no longer exists
build skip ci
making the docs a little clearer
show items in lexer data
default top to true
fix format log compatible with array
update bracket highlight after document change too
contentrect passed on safari is broken when zoomed
fix 修复sockettask close不传参数报错的bug
fix umd global
pass custom options from save to model exists
remove unused property
add paste and match style feature
docs fix description for tiledobject point
add check again
update render backtracking tests for remove first usage stack trace
style fix lint
fix constants duplicate declaration
test fix tests
fix main engine client add catch
yarn prettier all
refactor aws http api unify warning messages
feat fetch task item peers api
fix inneraudiocontext 移除初始化无效src
add knowledge of react is typeof to rollup
fix inline token error
fix 撤销 audio src
increase test timeout
fix 去除 uni uploadfile 验证 file 参数
quickapp 更新 manifest subpackages 判断
fix moved into src file
refactor quickapp quickapp vue
merge branch dev of https github com dcloudio uni app into dev quickapp
fix 参数验证不支持二进制数据的bug request data 为 array报错的bug
fix cli converting file uniapp cli js to unix format
refactor components remove check that was put into components logic
add promise reject to importer error cases no issue
fix container pointtocontainer function
fix input enabledebug function add debug shape to parentcontainer
fix 检测到 pages json condition 起始页无效时终止编译 输出详细信息
chore skip flaky test
set the property on custom elements
docs add options strict to model bulkwrite
docs improve options strict description
style fix lint
add tests for our cases
feat bt listen ports config
remove unnesseary call to textinput destroy
merge branch master into development
docs lib model fix punctuation
add test for key
treat escape token same way we treat plain text tokens to allow their transformation in renderer
fix mp alipay getplatformglobal
fix quickapp request post data参数错误问题
fix ios端 uni onbleconnectionstatechange不触发的bug
fix get random int util
fix independent ttw calls in physics arcade world js
docs add skipid to model documentation page
javascript hint add options additionalcontext property names as hints if not hinting within context
fix denyallow thrown away with pure hostname patterns
update package labels in tachometer config
fix typo in testflags copypasta when setting up old and new aliases old should mean that enablenewreconciler is off
fix 修复部分平台使用 css var 编译报错的问题
dont close dev server when tests finish
returned error upon oembed catch no issue
refactor components 虚拟列表的 raf 和 caf 从 runtime 中载入
add embedded language formatting to playground
update showactionsheet 支持国际化
chore cli add message
changed object destructing to simple assignment
fix linter s errors
fix chooseimage path
chore check js in title in lint changelog js
rust add some keywords
feat batch remove task api
add src noconflict ace to typescript module declarations
disable node no missing require for at glimmer debug
test selection specified in query overwrites option in schema
revert using passed value on exclusion to projection
add marked use
add more code comments
docs api detail what the insertmany promise resolves to
merge branch master of https github com typescript cheatsheets react typescript cheatsheet
set texturetintpipeline on each postrendercamera call
highlight code blocks in excerpt too
add algolia config
fix per project team email to parsing issue with empty doc security tag
skip locales in dev
only read dom attributes if hydrating from excess dom
show hint make linter happy
perf to save the contract
style after lint
test perf torenderexact method
style after lint
fix incorrect compare rule
style after lint
fix dev build html webpack plugin
add down key to types for smoothedkeycontrolconfig
chore configure aws sqs commit message namespace
fix 优化 picker 通讯
docs correct type for canvastexture getpixels
docs phaser physics arcade components friction
feat cli add uni automator
fix chooseimage tempfilepaths
enable no useless escape rule
docs add options path for model populate
docs add string as an option to model populate
fix engine shutdown
get rid of event listener warning when running tests
update outbound link
remove duplicated global declaration
disable no prototype builtins rule
add link to issue
fix noaudiosound destroy
docs noaudiosound does not extend basesound strictly speaking
add boolean variable
use native dataset if available fixes typeerror in strict mode dataset is readonly
fix failing test
refactor use instead of null in loop and use for of
refactor convert last loop to for of
apply arguments instead of passing an array
chore configure aws stream commit message scope
mind possibly too long abbreviation
merge branch develop of github com mermaid js mermaid into develop
perf diff string
fix weird test issue
export qunit only
docs arcade physics collisions
do not run tests twice in travis
enable no prototype builtins linting rule
fix types add missing field action select option option value value label label
remove duplicate cacheoptions
test use model validate instead of document validate
test verify model validate returns errors for multiple elements in array
test only remove one space
add newline to space test
remove unused strings
remove extra space kiloreux was the space intended here what do you think
fix 解决部分情况小程序判断 markerid 报错的问题
fix schematype treat select null or select undefined as not specified
test aws websocket resend messsage if reception was not confirmed
cast year input to int
locale update bg js fix cardinal noun sigh
perf cache loadlocale misses to avoid fs ops
docs math rotatearound rotatearounddistance
docs generic return
locale sl improve date format
add week relative time threshold
locale hr improvements on relative time
test bump test timeout seems appveyor tests are too slow to execute
test force click finish install as there seems to be a bug in cypress
remove swig rendering condition
chore add test android
remove unnecessary strict false from test
docs correct tilemap renderdebugfull
docs model add options overwrite to findoneandupdate and findbyidandupdate
add note for replaceone as an alternative
locale ka fix future relativetime altering unrelated strings
perf get match indent to code
clean up some comments
vim bindings remove a duplicate entry in defaultexcommandmap
feat app nvue scrollindicator
feat app plus backgroundcolor background backgroundcolortop
docs model add findoneandreplace as a more relevant alternative to overwrite
remove unused code from md plugin
fix schema split path name when validating if it s reserved
fix spelling mistake
chore update showmodal
docs model adds options limit to model insertmany
update size limit to reflect terser downgrade
fix model verify discriminator existence before trying to access it
updated canary admin site endpoint temporarily these changes are not ready for prime time
shouldn t be part of the container
chore register aws info commit message scope
fix linting error
test aws websocket stop sending messages on error
enabled unkeyed focus tests
fix data type
update layerdataconfig js
fix method name
add more docs
update inputplugin js
perf throw error if raw undefined
add node environment to sentry metadata no issue this allows us to differentiate between development and production
fix connection make connection returned from usedb inherit config from default connection
update input js
update select js
add form to the props type definition
feat automator add json config
update hash navigation js
chore disable gql linting
ci disable thread loader on circleci
refactor getindex move collectindexes definition after usage
fix model allow add desc index on schema
changed bulk email error to emailerror no issue this error is more suitable than the generic ghosterror
fix wake when running should not sleep
fix mp windows
test assert string descendeing index works
skip logbox tests to fix ci
add select all and redo features for text editing on macos
test fix usage of assert throws for older node versions
allow mongodb options in distinct
locale fil fix metadata comment
avoid new connection config mutating original config
docs add flattenmaps and aliases to document toobject
don t build react data for www
build flight runtime for www revert part of earlier change
cleaned up some weird require paths these require paths go back too far they work but it s odd
fix docs use mongoose model instead of this model in hooks
style fix lint
correct types for gameobject body
fix query query select field false should not overwrite schema selection options
prove that test fails
fix model fix throwing error when populating virtual path defined on child discriminator
add fix again for virtual options options access
fast path for overlaponly
refactor aws http api fix wording of error message
chore improve commitlint configuration no need to validate scope casing with prefixed scopes
locale el remove dependency to util
fixes channel name issue
refactor aws cloudformation improve conditional logic
update karma conf js use the recommended karma browserstack launcher settings
ruby symbols precedance before shorthand conditions
skip intersects check by argument
avoid fractional pixel in viewport height fractional pixel could lead to minor scrollbar glitch reported internally
always search the static tree
docs phaser physics arcade world usetree static bodies are always kept in an rtree
test temporary debug patch for randomly failing test
update generate js
tus use hasproperty util to make this if condition more readable
docs correct type for phaser loader file url
refactor queryhelpers make exclude depend on falsy values
docs for phaser physics arcade body facing
fix greedy string highlighter pattern
fix undefined key in error message
fix installing scoped cordova plugins
chore register aws cloudfront commit message scope
update 注释调试 vdsyntc 日志
update framework class reference to component
test against window innerwidth to toggle vertical layout
wait a bit longer before testing viewport width
fix installing cordova plugin from git url
fix 解决一般情况下优先使用 easy com 组件的问题
style fix few stylings
chore fix failing tests due to migration to the jest
build fix type cast
fix generator avoid generating route with extension
print texture key in warning for zero frames
docs correct description in arcadeworldconfig
fix only cancel cancelable touchmove events
docs document copy model save to document save
tcl mode add as a linecomment
fix 允许app端uni settabbarstyle传入其他选项
fix use pure from static folder
style fix lint
debug add debugging logs
ensure that the defines build target constants in gulpfile js cannot be changed
test model fix flaky text for model populate
fix casterror add kind to cast errors thrown by query execution
fix standalone ensure pkg bug workaround is applied on windows
doesn t have effect on stoppropagation
zoom check for cancelable event
merge branch master of github com transloadit uppy
add children after configuration ensures the create handler is called
don t discard config
add missing config when passing children only
fix position from scaled sprite
fix remove debugging logs
make the default undefined
feat cli compatible with subpackages
avoid bad scrollbar replacement into width values
fix 修复 app 端 cover 组件高度显示错误的问题
chore fix typo
dart mode add late and required keywords null and never types
refactor version use package set method to assign package version
refactor project internal default export named
refactor project extract glob option logic into helper
refactor project don t mutate packageconfigs parameter
refactor project inline array flattener
fix uni settabbarstyle 增加 backgroundimage backgroundrepeat 校验
update documentaion links
fix 修正 ios tabbar 高度获取
test aws http api bring back testing logs as issue on aws side was fix
fix model allow pojos as schemas for model discriminator
mode meta remove unwanted space in tiddlywiki mode name
test add customparseformat plugin month function locale test
fix npm build clean script
refactor engine instance listeners
chore disable electron security warnings
workaround for gboard converting two spaces into dot
test cli test singular config params
style fix lint issues
fix base allow setting strictquery globally
fix remove debug line
style fix lint
fix vue app catch error on full static for store mutations release
style fix lint
update returns updatewriteopresult
chore fix ci
chore always reset webpack stub
docs fix docs omission from path
docs model findbyidandupdate new param fix
regression remove avatar provider temporary
fix tray destroy remove listener
force ci build
fix lint errors
remove log and verify lint
fix too fast to shut down the engine
locale ru simplify regex only need to be escaped so remove redundant before
fix ios vue页面无回弹效果问题
eslint config base patch include context exception for no param reassign
tests document add github issue number
remove old errorcomponent global
fix webpack copy plugin path
test ensure test is not immune to race condition we observe occasional fails due to differently sorted results
fix addstylesclient 去除 flags s
fix slide notes printing
fix in pagination detect uniquenavelements
fix error for missing audio tag already released
fix switching system theme will not auto switch app theme
fix cli ignore id class style inline template bool attr
ensure lack of super in init assertion is helpful for native classes co authored by robert jackson me rwjblue com
chore update saucelabs browsers
update swiper js
fix code error
move to static class fields
remove unused context param
add debugging code to create debian package js
add debugging code to create rpm package js
fix error for headless renderer
refactor improve macos fullscreen mode usability
refactor time constants
revert accidental change
lint document remove impossible to execute branch
lint compile fix no setter return
enhancement connection make server selection error less deterministic
add phaser gameobjects group setname
remove spurious assignment
examples add transloadit debug log to dev example
test remove path parameter
update build js js
docs depth depth and depth setdepth
add phaser gameobjects group setactive
update component js
remove parsemessage function
fix jqlite apply suggestions from code review co authored by michael prentice splaktar gmail com
fix jqlite improve documentation
logging deactivate debug log in production
email make hooksend public
refactor rename auto hide window
mention logging to writable sreams
scale parameter for coverflow effect add
exclude timed out links from logs
add explicit test for textarea defaultvalue
fix parsing error
fix parsing error
remove debugging code
remove debugging code
fix typo in gruntfile no issue
don t chmod chrome sandbox
style fix lint errors
add info to the scene manager method headers where data is passed to scene init and scene create
add a few assertions before taking sudoku snapshots
gfm mode allow forward slash in fenced code language
deleted an extra line while remove comments
fix return type in text gettextmetrics object phaser types gameobjects text textmetric
enable webview support with the correct option
script refactor log strings for readability in script lib verify machine requirements js
docs document add validatemodifiedonly to document save document validate and document validatesync
normalize english australia ocker
docs add missing phaser gameobjects shape geom
update eslintrc js
add slide width height css variables
docs phaser tweens tween callbacks tween stop does not invoke oncomplete
fix bug about setting xradius yradius by width height xradius yradius is half of width height
dom add styles method
core virttual fix before after vars
docs ngrepeat missing closing backtick
clean package before release
style fix lint
remove trailing whitespace
remove unused mochacli variable no issue
fix umd export
update release js
fix typings build
fix package cleanup
mousewheel fix forcetoaxis inverts scrolling
remove unused modular instance
keyboard fix a typo chromedriver mksnapshot
script use let not var in verifypython
fix 解决 ios uni showtoast 部分情况文本显示不全的问题
corrected a typo in the jsdoc
build app view
feat document add document getchanges
eslint config patch set explicitspread to ignore for react jsx props no spreading
texteditorcomponent didfocushiddeninput no zeroing
texteditorcomponent hiddeninput preventscroll
updated net level determination
fix new task uri
eslint config base minor disallow multiple empty lines
unnecessary directory check
vbscript add missing keyword operators eqv imp and is
vbscript add missing me keyword
vbscript add missing separator
vbscript add missing msgboxstyle constants
vbscript remove redundant asp specific end of comment lookup
vbscript make white space matches greedy and ensure that the statement ends with boundary
vbscript add missing operators integer division and exponentiation
vbscript add missing match alternation character in the pattern
vbscript add missing built in regexp object support
vbscript follow common tokens guideline make me keyword a variable language
vbscript add missing keywords preserve erase execute and executeglobal
vbscript add missing tristate constant vbusedefault
vbscript add missing msgboxresult constants
vbscript preserve official title and camel cases in identifiers for the sake of readability
using withfiletypes to filter out non folders no need to use path basename
locales fix newline
adds message for when mongo server is not starting on windows
fix regex and fix incorrect test
updated copy for email address change template no issue fixes typo and title for email address update mails
workaround for chrome returning browser zoom as css zoom on documentelement
add tripple quote string support to swift mode
hooks also check for existence of raf while cancelling it
markdown mode change fencedcodeblockdefaultmode to text plain instead of
set default random seed for finance tests
feat cli add npm scripts
julia mode tweak indent method
update random seed value for finance test again
add random seed for time test
docstrings layer s removefrom works on layergroups too
fix lint error
fix uploadtoserver can t be null
update webdriverio config to match the new schema
remove deprecated screen api from the cache
fix lint errors
fix linter on cast array filters test
fix regression tests
update tokenizer js
hotfx vue app app is not defined release
chore ci temporary disable dev cli test release
chore one more missing done
add catch and error logging
core fix onresize handler
merge branch master into next
wast mode update to match latest webassembly specification also shorten the regex for the keyword tokens a bit
test fix tests
fix fix tests
style fix lint
docs specify the array field syntax for invalidate
docs array field notation for invalidate
refactor move areanalyticsdisabled to analytics aredisabled
refactor aws deploy refactor to bbpromise try
fix no shadow linter warnings in image size js
docs correct timereventconfig startat default
update lexer spec js
update lexer spec js
update lexer spec js
update lexer spec js
update lexer spec js
update textstyle to have jsdocs for wordwrapwidth wordwrapuseadvanced wordwrapcallbackscope wordwrapcallback
add uppy dragdrop input class name back to fix tests and potential backwards compat customizations
collapse js return early
remove a couple of variables used once
update src tokenizer js co authored by tony brix tony brix ninja
update src rules js co authored by tony brix tony brix ninja
update cjs js
script update comment in verifypython
script refactor variable use in verifypython
docs minor english fix in validation
docs validation add validatebeforesave and invalidate
fix hidetoast errmsg
fix an incorrect reference to an outer variable actor when processing actor widths
runmode node addon export countcolumn again
remove redundant regex symbols found a few more cases of redundant symbols in addition to fixes suggested during review
fix errant only in tests
fix generator rename export hook to distremoved
merge branch dev of https github com alvarotrigo fullpage js into dev
add test for crlf newlines
refactor cli preprocess
test do not remove service on fail
fix width of notes left of and right of
yarn docs typescript formatted
style fix lint
feat dockmanager add handlespeedchange
fix model fix conflicting setoninsert default values with update values in bulkwrite
allow non null falsy values for setdefaultsoninsert
fix autoplay bound context
regression admin uploading foreign profile image changes own image
return needed objects
test fix tests
used current branch
fix 安卓 nvue 页面 promise 增加 finally 支持
update lib plugins helper tagcloud js co authored by sukka isukkaw gmail com
add keyboardplugin removeallkeys
add git config
make gitgraph commit ids out of hexadecimal chars
make user agent deterministic
test update async config size limit
corrected the test case description
react fix autoplay param name
removes versionsfrom of packages url package js
fix orthogonal world to tile
fix core only warn for duplicate module if metakey is not specified
remove unused arguments from applyschematypetransforms
refactor improve cli notification border
missed activationwidth new configuration option for sequence diagrams
apply suggestions from code review co authored by william reiske wreiske mieweb com
feat wxs add getcomputedstyle
fix logging log database only in relevant commands
fix npm typo
update 优化 getsubnvuebyid 错误提示
perf avoid running plugins in clean command
refactor standalone fix cli help
update schema js
sas mode add missing keywords i ve noticed these few are missing all from proc mixed
add test for hydrate callback
fix tag code parse wrap option
merge remote tracking branch origin develop into release candidate
chore remove message about processes not saved
update reporting 引号
math is hard sometimes
style fix lint
update remove error report
style fix lint
test do not attempt to remove not created service
test optimize resources resolution in cleanup
chore quick copy change
test fix tests
docs model fix typo
treat new undefined as not mentioned on findoneandupdate
fix unrestricted any character for redos and remove redundant unused capture group
fix for broken build issue with config use in flowcahart and gantt diagram
cleaned up onvalidate method on settings model no issue explicit arguments and async await make this nicer on the eyes
fix backward compatibility fix for tarball modules
refactor improve var name
docs model add note to use global option returnoriginal
correct prev and prevframe
minor code review re element picker avoid potentially add element twice to the resultset
or just simplify by combining both conditions
test fix tests
update stylecompositionexample js
fix passing a null ref
do not break negative lookbehind regexp group
update stylecompositionexample js
update 优化编译器提示 pages json page对象配置错误
fix audiocontext 移除 stop 方法
fix rename audio js to inner audio js
fix express jwt config no issue express jwt expects an array of valid algorithms not a single algorithm
rename packagepaths to packagenames co authored by sadick sadickjunior gmail com
enable in place builds for full rebuilds on windows
refactor server externalswhitelist externalsallowlist
make strend const co authored by tony brix tony brix ninja
make emend const co authored by tony brix tony brix ninja
fix running zodern hot locally
fix clearing reify cache
add dispose handlers and hot data
fix error when unable to resolve module
fix sending remove files list to client
fix fast refresh error when hmr is not available
don t throw error when comparing options w wo collation
update containerfactory js given that container s constructor allows x and y to be optional the jsdoc should match that
fix fix typos that broke tests
refactor cleanup condition
fix throw operational error as operational
style cli improvements
refactor cli improve help resolution documentation
fix cli show interactive help unconditionally on help interactive
fix cli show version info unconditionally on v or version
fix path moveto optional y param
remove textarea branch
update compat test browser textarea test js co authored by jason miller developit users noreply github com
join template literals
tooltip js remove variable used once
scrollspy js minor refactoring
fix failing test from bad develop merge
fix browser error snapshots
update 移除历史版本无用代码 修复支付宝静态编译检查错误
test throw when events in question are not found
refactor improve setting readability
test aws stream increase log confirmation timeout
update packages uppy dashboard src index js co authored by renée kooi renee kooi me
fix 修复部分情况 css 编译报错的问题
fix data type
fix mp remove sharedobject
fix android 偶发字体变小问题
add maxgputextures property
update config js
add assertion to existing test co authored by tony brix tony brix ninja
ci ensure to cover serverless enterprise plugin under canary releases
fix core interceptor
fix 纯nvue项目无法加载本地 hybrid html
fix mp proxy
fix audio oncanplay事件失效
refactor improve var naming
fix ensure deprecation logs support mute settings from service config
fix ensure dependencies resolve correctly
don t build shallow renderer umd bundles
update webglpipeline js
feat wxs support preventdefault stoppropagation
fix migrator fix help message
fix stat 修复微信小程序未配置安全域名导致请求报错问题
add missing jsdocs
add setgameobject method and currentunit property
mesh and quad game objects now support rendering with normal maps
update forwarddiffuselightpipeline js
if normal map set advance the current active texture
update webglrenderer js
update textwebglrenderer js
fix quiet the logging
lightsmanager cull now takes the viewport height from the renderer instead of the game config
update dynamictilemaplayerwebglrenderer js
update inject js
always set rotation
statictilemaplayer now includes the modelviewprojection mixin so it doesn t need to modify the pipeline during rendering
enable the pipelines again
update ropewebglrenderer js
remove commented out console log
testing rendertextures currently broken will fix next week
test fix tests
chore cleaning up spec
update test case
image editor can t edit remote files
update config js
enable slot appinput spec
refactor set returnoriginal instead of new with global returnoriginal option
docs returnoriginal clarify that returnoriginal doesn t set new and link to findoneandupdate tutorial
fix nvue map api movetolocation 参数经纬度无效及无回调问题
fix typos in webhook error message no issue
add failing test for useragent proving it is not deterministic and does not respect random seed
avoid returning a value when injected as content script
fix cli latest
test validate fixture extend args
adjusting initial classdiagram shape rendering
replaced remove nock method no issue
chore remove extra poolsize option in comment
remove debug line
remove debug line
fix 适配支付宝小程序api getsavedfilelist
add missing require import
test fix zip permissions tests
updated variable name to cnf
fix if theme inclusion for classdiagrams
fix cli vue shared
test fix buffer usage
fix cli normalize runtime path
merge branch mermaid js develop into develop
build fix source map generation
mousewheel add threshold for wheel scroll event
update rendertexturewebglrenderer js
fix make paths plugin silent
update rendertexturewebglrenderer js
update rendertexture js
add threshold param for allowed time between wheel scrolls
merge branch master of github com automattic mongoose
chore register aws schedule commit message scope
types allow text setpadding to receive an object
testing the label width in percy
fix eslint issues
test aws local invocation improve test reliability
test introduce debug log
fix fix aws sdk workaround it should not be applied if nslookup command is not found
fix a new config issue diagrams now properly reset to siteconfig when there are multiple diagrams on a single page
fix use core js package json to read version
style fix lint
improves the message on cache build option when there is no valid cache build
fix aws credentials improve credentials error recognition
chore recognize aws credentials commit message scope
test fix test title
test fix async tests setup
fix nvue页面relaunch到vue页面时plus setstatusbarstyle对象不对 导致设置状态栏无效
docs generic return for add existing
fix 解决支付宝平台v for中自定义事件内使用复杂表达式报错的问题
fix call to getbitmaptextsize
floor drawfillrect values
update bitmaptextwebglrenderer js
update config js
update bitmaptextcharacter js
fix bounds sizes
update bitmaptext js
fix set versionfunctions to true only in aws provider case
refactor simplify condition notation
refactor remove obsolete null assignment
avoid doing unnecessary work in annotationlayerbuilder render such as cloning a pageviewport when no annotations exist for a page
remove un used properties
remove un used properties
bitmaptext predestroy is a new method that will tidy up all of the bitmaptext data during object destruction
simplify param names
core install modules in core class
if you give an invalid bitmap font key the bitmap text object will now issue a console warn
correctly type style in textfactory
use textpadding type in text game object
log suite start failures in the test runner
remove deprecated events
fix the type of svggraphics
expose new function
use typedef to define the type of globalworkeroptions
add types to functions exported as api in src pdf js
tweak for the type of pageviewportparameters viewbox
always run in headless mode
fix afterframe import
include char top right and bottom values
get character using char top right and bottom
update bitmaptextcharacter js
update bitmaptextsize js
refactor config schema for consistency mark required top level props
chore register config schema message scope
integration components life cycle fix removecomponent
fix incorrect types in the api documentation
merge branch dev into alpha
fix the type definition of typedarray
merge branch dev into alpha
fix char right adjustment
检查 set cookie 大小写
don t add frame if empty
no more innerhtml in fontmanager js
no more innerhtml in hybridrenderer js
no more innerhtml in svgrenderer js
no more innerhtml in canvasrenderer js
eslint config base add id denylist rule
fix fix missing s in directory path
chore remove extraneous exclamation marks
reset all textures on rebind which is what externs need when finishing
fix the type of pdfdocumentloadingtask destroy
remove common pr title prefixes
test prevent cleanup on test failure to provide debugging means
fix cli mark help as command that doesn t depend on external plugins
chore register dashboard message scope
add component stack to prop type validation error
fix interact init
indent a command
ensure eslint plugins use the correct tsconfig json
fix support react virtual rtl direction
update parsetilelayers js
bad bad idea
corrected undefind typo in deprecated markdown package
prevent grunt from exiting without letting express stop
allow to pass font in textstyle
test update test
update setpadding jsdoc
fix change and to persian equivalent
chore tweak algolia instantsearch component
block the percy snapshot until the collapsible animation has finished
enabled batched members import method through enabledeveloperexperiments flag no issue allows early testing of batched import method
small grammar fix and addition of units to output
temporarily disable danger spellcheck until we can figure out our own settings
fix bug cannot break line well co authored by tony brix tony brix ninja
fix link to pull
fix www service worker unregistration
fix app平台 app vue 生命周期 onshow 无参数问题
fix parameter naming for members importer
add failing test when maxlength null
fix typo in comment
replace null to undefined on normalizing props
fix for broken build
fix 修复小程序 v for 中嵌套复杂事件表达式编译报错的问题
remove illegal logging
fix cases of illegal invocation with nowoif
use arial in statediagram rendering tests
wow this spellcheck is so brittle
fine tune nowoif scriptlet
fix code style
update 使用过时api uni onuistylechange时 控制台提示用户使用新api uni onthemechange
further fine tune nowoif scriptlet
fix module not found test
expose new function
update getraysfrompointtopolygon js
missing semi colon
fix config schema recognize catch all pattern in disableddeprecations
double waitsfor timeout value
fix webpack config problem caused by breaking changes of worker loader
fix 修复复杂的事件表达式中含有 undefined 时 在小程序编译报错的问题
remove false import
refactor show variables resolution info less frequently to not pollute eventual mfa input
fix aws http api recognize api id passed as cf import instruction
chore configure packaging commit message scope
drop unused import
drop focus on test
fix typo in name inside manufacturer js
fix postcss add tag editor
fix keyname function for cmd key corrects return value expected mod instead of cmd mod
recognise cmd key in firefox
return the query with the findcontrols
fix config schema fix required properties configuration
remove console log
feat highlight parse caption option to prismhighlight
add more test cases for countsetbits
use new utils function
chore remove clear site date
refactor config schema rename cfimport definition to awscfimport
remove redundant consts
update webglrenderer js
docs add note about using ref as function
rename fixture suite
add discord to page
improve first names
don t include gh pages task by default
fixes default value issue for example page
add simple fullscreen config
addresses review comment
mibor better variable name
expose the new events
processqueue checkqueue is a new boolean property that will make sure only unique objects are add to the process queue
update particlemanagercreator js
test ensure result of tests does not depend on package version
tweaked rt draw fix
when using the gameobjectcreator for containers you can now specify the children property in the configuration object
use performance instead of date for measuring package load time
test ensure debug logs in integration test deployments
test improve error handling
fine tune collapsed sections behavior
test custom publicpath for full static fixture
refactor standalone improve upload confirmation logs
test aws lambda fix propagation time allowance
test fix test errors
fix typo in error message thrown by unimplemented createindex
allow generics support for class title definition
remove check for ssl when applying serveroptions
fix for class title label css
reset flags during reset
update initialize js
only collect coverage for pull requests
remove usedebugvalue for prod
remove unused var
test ensure long stack traces
fix bad code
test aws lambda improve error handling in efs tests
fix documentation for mongoose query get
remove browser code from server bundles
add length check for deps array
add test to codify cgs injection order as part of the api
chore configure analytics commit message scope
adjust minification mechanism
fix typo require required
ensure port instance exists when waiting for connection requests
staticfilteringparser analyzeextra has no argument
fix update dutch nl locale to set correct yearstart
remove protected status
test aws lambda recognize efsioexception error as recoverable
refactor automatically align multiline deprecations
refactor cli remove no longer valid comment
react native fix lint
fix jsdoc event names
updated jsdocs for play to show it can take the new config
fix benchmark wrong measurement on hot processing
default rotation as well
chore fix indentation
fix protocol altitude type
fix typo require required
test aws stream ensure update inserts on repeat
test rely on external wait util
fix measuretext arguments
add arcadeworldconfig fixedstep
remove code for unused message
actions setorigin will now call updatedisplayorigin on the items array otherwise the effects can t be seen when rendering
fix data type
support repeat count for stopping
use new method name and parameter order
tidy up order
refactor prevent is local detection on locally installed
update buildgameobjectanimation js
fix regex for mainlink
update regex checks for source code demo clients
prevent error when is in the name
set default for react fast refresh globals
chore sort car manufacturers alphabetically
chore add car manufacturer
revert request update in http package
update mermaidapi js
fix creating log folder in sync mode
update animationstate js
update animation js
because do while loops are quite dangerous
fix for legacy class diagram title for class bold styling
use if construct instead of conditional assignment
build fix overeager regex in website examples build
no update unless playing
fix strong overlapskip
fixup chore some trivial major version bumps for dependencies
tools list command add mising local entry to pacakge entry in json mode
docs correct type for body customboundsrectangle
test rename module to match conventions
test aws sns fix multiple result batch handling
chore fix eslint override setup
refactor config schema improve signal intent comment
use the new pipeline manager methods
update pipelinemanager js
no need for gl ref
adds support for both quotes in the ts side
use a regex for injecting vs editor editor api as a dependency
reset cursor position when focus is out of text field
fix config schema recognize condition on resource configuration
test aws http api fix after drop for timeout support
test packaging ensure test is deterministic
merge branch master of https github com photonstorm phaser
remove inputmousecapture and add configs for inputmouseprventdefaultdown up and move instead
bump version to use the latest version of minifier css
object values may fail for unknown reasons work around this issue by using more reliable object keys
says webglrender but it is in pipelinemanager
use loose equality for prop comparison
go with the null undefined coercion instead
change status transloadit com to the more isolated transloaditstatus com
merge branch master of https github com photonstorm phaser
the camera prerender method no longer receives or uses the resolution parameter
remove camera resolution use
remove camera resolution use
the canvasrenderer no longer reads or uses the game config resolution property
the phaser scale events resize event no longer sends the resolution as a parameter
revert canvas properly restore all the remaining items in statestack in enddrawing
allow setting nonce in browser policy content
fix double login on enter in accounts ui unstyled
fix refresh on using forgotpassword
merge branch master of https github com photonstorm phaser
add debug draw
expose vertex and face
replace n and r by n and r when saving a string
add setposition and translate methods
don t use temp track cleanup process fails when the file is locked in parallel tests
size tweak for renderqueue
update fs tests
fix config schema recognize string value at dependson
udpate android showactionsheet 增加取消按钮和微信保持一致
new method signature order
fix assertion to pass on windows
fix linting errors
feat mp match media uni match media
feat match media uni match media
add ability to use custom easing function s with particle emitters
updated path ref
update test js fixes wrong channels counter
fix app 端 measuretext 支持同步调用
fix for broken flowchart test cases
use the projected coordinates
let them define the orientation
remove duplicates from log
make defaults match jsdocs
eu domain suffix in nl locale
nl locale company suffixes
add hacker namespace to nl locale
filter category by name
add onworldbounds arg in setcollideworldbounds
docs arcade body setcollideworldbounds
fix error when no reloadable parents
fix 优化 uni getimageinfo 失败返回信息
docs correct cursorkeys properties not optional
balanced demo org chart
update codicons path
fix mp platform mediaqueryobserver no option
fix dynamic layer
fix group getmatching
trying to run browserstack inside circle ci
enhancement document pass do document to required validator
trying to run browserstack inside circle ci
trying to run browserstack inside circle ci
trying to run browserstack inside circle ci
trying to run browserstack inside circle ci
trying to run browserstack inside circle ci
trying to run browserstack inside circle ci
remove unused property
add lexer lexinline
add parser parseinline
test marked parseinline
remove focused tests
refactor config schema do not pass obsolete arguments
add reserve property to particle emitter config
improve types for particle emitter deathzone and emitzone
minor code review of subscriber scriptlet
clean the mesh after rendering
remove console log
not yet fix but no longer crashes build
update the camera if dirty
add per model cull mode
add destroy method
add layer property
updated missing docs
corrected comment describing redirect behavior no issue the redirect is done from feed to rss
release script allow preparing rc from npm
add setposition and setscale
explain render modes better
fix breaking change fix lint log fix breaking change
remove unused exports
mirror fix in jsx runtime
should not keep ref in props
remove useless change
update jsdoc types
remove duplicate code
fix lint issue
remove duplicate code
format version info output
delete version options
update focus trap usage
tools list fix unwanted pretty print of json output
fix jsx runtime not published
further fine tune new aost scriptlet
do not syntax highlight in my rules pane
fix property name
fix slidesperview type
vue clone slides
fix module resolve error
fix csssupports 在非chromium内核的edge浏览器中不准确的问题
refactor print read provider values from provider
chore register print commit message scope
chore register aws msk commit message scope
fix classname normalization
test aws cognito improve logs handling
tidy up docs
fix test add timeout
fix test upload test utils
fix issue with re open closed prs
found the fix
fix classname normalization when props is null
fix scrollto coordinates issue
docs arcade body reset only game object receives the passed coordinates
docs arcade body setoffset
remove updatecenter in setoffset
fix drawing group to rendertexture group children generic gameobjects expect a camera argument in their willrender function
merge branch dev into alpha
exclude test folders from packaged app
exclude specific spec folders picked by hand to make sure they are test
fix wording src packages styled components base js
test test locale names are in lower case
merge branch master of https github com photonstorm phaser
fix aws alb ensure to treat provider alb authorizers as optional
update help js
fix color parsing
validation check on if wrapper exist
fix shake effect
add to updatelist if not on display list
allows developers to disable the check for the same origin
add missing semicolons to appease eslint
container doesn t need to be on the updatelist so remove the handler
fix return states
renamed method to setfromeuler
add projectviewmatrix and unprojectviewmatrix
update gameobjectfactory js
update renderer js
allows developers to use redirecturl on loginstyle popup as well
account for pr remarks
refactor config schema remove dead configuration
feat coerce input configuration values
refactor config schema improve comments
apply suggestions from code review co authored by jonathan haines jonno haines gmail com
update help js
update center after separation
don t update center before separation should be unnecessary now
tilemap hexsidelength is a new property that holds the length of the hexagon sides if using hexagonal tilemaps
fix config schema allow both arn and topicname in sns event
merge branch master of github com gorhill ublock
group now extends eventemitter allowing you to emit custom events from within a group
merge branch devel of https github com meteor meteor into chore modernize standard minifier css
updated server js
fix wrong naming of first name variables in the norwegian locale
update center after world bounds collision
emit error event for errors handled by an error boundary
modernize reload package improves test
test document correct test title
test fix tests
add check for the wrapper on destroy
chore get rid of unnecessary log statement
don t set outlinestyle if undefined
test improve test reliability
add the objfile type
test aws cognito improve tests setup
update vertex js
add z parameter
add cameraz parameter
pass in the z value
transform based on z value
clarify the context under which log file is used
merge branch master into develop
create materials container
expose new functions
fix property name
test reorder info retrieval
javascript mode fix potentially exponential regexp
add setuvs method
update generategridverts js
core add new ipads sizes
fix android measuretext
fix sourceurl for replaced modules
golf some bytes
add escape slash to cell matcher
allow use of old method calls for now
test aws local invocation increase error margin
fix config schema fix api gateway authorizer schema
bump version number
fix default value for onjsoutputfiles
fix fetch when running in electron webview
fix update catalan ca locale ordinal
update 去掉编译到头条提示语 测试可以使用 data 作为属性保留名 不允许在自定义组件 scoped slots default 中定义为 props
update webpack config js
fix 回退 uni savefile 文件命名规则 避免存在特殊符号导致的部分问题
avoid introducing breaking change
fix config schema fix provider apikeys required properties config
fix config schema fix provider apikeys schema
fix creation of emailbatch and emailrecipient collections no issue
handle negative scales
fix config schema do not mark layers path as required it is not required when package artifact is passed
fix updating modules that haven t been imported
fix using hmr with dynamic imports
use mesh size
update collider js
update world js
update setcollisionbetween js
update gettileat js
update gettileswithinworldxy js
avoid deep diving into loads of functions
ad 全屏视频 移除关闭后预载下一条数据
bump package standard minifier css version to beta
fix handling updates to dynamic imports with hmr
add macos read and write find pasteboard methods
fix config schema fix aws stream event consumer schema
testing if we can just pass off rendering directly
update spinegameobjectwebglrenderer js
update webglrenderer js
remove interpolationpercentage parameter
update spinegameobjectcanvasrenderer js
add mention about masks to docs
mind whitespaces when double clicking to select filter option value
don t add white space when measure last word
fix 配置 uni getimageinfo 临时文件路径
fix touchevent remove unused check
don t check skeleton alpha use flags instead
add face alpha getter and setter
add mesh ignoredirtycache flag
use new shader property
update pointlight frag js
remove focussed test
resolve promise and then rerender
fix bad tertiary operator this fixes the tertiary operator used to compute the edgelabelbackground theme variable
update theme base js
update theme base js
container revert max width change for xs
format log json stringify add second parameter
feat 鼠标事件支持 pagex pagey clientx clienty
add webassembly to meta
fix no triggerevent function when invoke this emit in page
eslint config base no restricted globals add better messages
style fix lint
ensure no large element styling is remove
fix more linter related issues
fix app uni choosevideo 使用非系统相册
remove debug code
update webglpipeline js
no need to reset during destroy
bind needs to return this
add max size value
shader needs attributes
move sequence so the renderer is clean
flush here rather than in renderer
testing new pipelinedata object
test fix fixture service config it only worked because all provider properties are mapped on provider
test improve flow error handling
fix app plus window left window right
the webgl utils getcomponentcount function has been remove as this is no longer required internally
pass the game object
fix connection avoid executing promise handler unless it s a function
stylus mode allow block comments
test fix logs
disable no var rule for cmd since it needs to run on all node versions
merge branch master into develop
feat app socket 支持 header
update webglpipeline js
fix config schema ensure to validate provider as in config file
fix incorrect count in built in benchmark
test improve error exposure
refactor standalone reorder requires
refactor analytics alphabetical order
fix config schema fix iam policy resource reference schema
remove all of the uniform code
cause a full bind on first set
parameter not needed
tidied up parameters
xml hint addon replace nested function with function expression
chore remove useless code changed assignment of this l variable in dayjs constructor
switch to xhr server herokuapp com endpoint
refactor config schema explain missing definition
fix config schema run validation logic unconditionally it s to ensure that properties normalization is applied unconditionally
refactor config schema no point in creating new object
refactor config schema run schema validation only in service context
merge branch master into develop
you can forcezero via the config now
fix incorrect reset value this could cause spurious error messages in dev console of content page
use const when value does not change
test refactor to async await
chore register aws iot commit message scope
chore cli add tips easycom
update graphicspipeline js
no longer requires temp matrices
fix mp weixin ref in scoped slots
fix analytics ensure to send payload when having all meta
chore register aws alexa commit message scope
call new unbind method
correct sampler id
add generatepath to api docs
don t replace non strings
add forcezero method and used it
fix wording on warning message derp
update meshwebglrenderer js
update singlepipeline js
update multipipeline js
fire remove mp weixin s aop of navigateto
fix a typo in updatesize
use new fbo stack
update webglpipelineconfig js
expose render target and webglshader
update rendertargetconfig js
feat analytics distinguish different standalone installations
empty parameter means reset pipeline
test clean up some flakey geojson tests
allow tilemaplayers to set post pipelines
set overlapr in separatecircle
fix create use main as default github branch
test helpers workaround ancient git client in ci
test helpers stop being nice about ci git garbage
test helpers clones should always have a dang main branch
make float the default type
test integration ensure checkout is from upstream of clone
test import run git setup sequentially parallelism messes up windows
test import run chdir sensitive operations in the correct order sequentially
postfx pipeline only needs a single quad
change to batchsize property
fix bitmap mask
hard code the topology
call rebind function
allow rope to have post pipeline
modernize reload fix typo in a comment
modernize reload reverts change in test code
add documentation for arrayfilters
fix mp wexin getopenereventchannel的tree sharking
last commit changes compiled format
fix picker use of extraneous body in suggested filter related feedback https www reddit com r ublockorigin comments jregqx
docs schemastringoptions add note about alternate spellings for minlength maxlength
expose utility pipeline
fix missing property
update postfxpipeline js
add snipcart to list of recipes
eslint config base patch fix spelling of than you mean than not then
fix test 修复console spec js在window下路径处理错误
soy mode add support for attribute forks the state at param def to detect as a type
doc update array filterby example to include all properties
testing new util methods
remove draw method
fix redirect uri on facebook oauth
merge branch fix redirect oauth facebook into devel
fix renamed function isnodeignorecomment isprettierignorecomment
chore cli add watch ignored
fix app plus 修复相册选图片二维码扫描时statusbarstyle显示错误
fix lint error
handled in batchquad method
return the freshly bound fbo
call the new camera methods
always pop and reset
feat schema make schema enum accept typescript enum objects
search js remove window location origin workaround
use string startswith
fix query maxtimems in count countdocuments distinct
update mermaidapi js
skip complex reorder test temp to view benchmarks
golf more bytes
only call placechild when needed
update mermaidapi js
update mermaidapi js
fix integrity tests no refs
udpated email template footer spacings no refs updated padding between ghost badge and rest of the footer
show the edit button only when uploadinprogressorcomplete
don t batch if no pipeline
don t forget to reset the viewport
fetch pr title github on undefined null as well as empty string
javascript lint addon add comment noting dependency add note on dependency on jshint js
auto boot and set ispostfx
update canvasrenderer js
update rendertexture js
clear post pipelines
add getpostpipeline method
destroy on reset
updated data types
update pipelinemanager js
update scenefileconfig js
update scenefile js
no longer use setpost
implemented reading webgl texture dimensions automatically when add them to the texturemanager
test expose fixturedata on runserverless result
enabled spellchecking for only mac
updated render target constructor
use single pipeline and force texture refresh
don t adjust viewport unless needed
trying to align drawframe
modified logger to have the user jobid
resize to match texture fbo
update dependents count
test fix jest args order
disabled personal tokens of inactive locked users no issue
add personal token removal when a user is deleted no issue
update rendertexture js
test should not use file protocol now that we have verdaccio
eslint config base patch arthmetic arithmetic
fix culled tiles return
add ability to skip post pipeline per object
force post pipeline skip
test yet another dirty fix for verdaccio testing
update simplify js
lint addon filter out duplicate messages on a single line
eslint config base add no nonoctal decimal escape rule
fix menushouldscrollintoview in menuportal example
fix avoid memory leak after installing a plugin too many times
replace integer with number
fix parsing of queryprune in static filtering parser
fix sans serif link style no refs fix regression that links in emails with sans serif setting wasn t underlined
updated post pipeline
fix triangletoline not checking start end point of line properly
chore add confirmation step when generating the changelog
merge branch master of https github com photonstorm phaser
fix emitter mask pipeline
add bindrendertarget method
fix postinstall hide message in ci
add jsdoc comments into package json lib
checking for null explicitly
converting to single defaultvalue
merge branch dev into alpha
split to multiple lines
add settime method
update animationmanager js
update texturemanager js
fix render texture crop
reset textures before draw
use the new depthlist property
fix 一键登录无 logout
update layercreator js
fix 修复使用 uni chooselocation 时摇树优化报错的问题
remove flipx argument
style fix lint
remove data key in base component
tooltip js reuse existent variable
extra check for existence of any aria label before overwriting it
ignore private functions
style fix lint
chore remove extra import
use strict equality checks when comparing vnode type
clojure mode fix exponential complexity regexp
fix cli webpack watch ignored
merge branch master into upgrade dev deps
fix misleading emailanalytics fetchlatest debug statement
test update font preload to use regex
add point light pipeline to the const
install point light pipeline build flag
update wdio remote conf js
fix delete unnecessary console
enable lights plugin again
update webglpipelineuniformsconfig js
tidying up the class
remove unused property
add on onactive hook
update main js
enhancement schematype pass documents to default functions as the first argument
update lightpipeline js
fix light distance and max test
add resetuniform method
no need for dirty property
fix aws api gateway fix integration schema
updated logging message for webhooks removes double space and tidies up the output
scheme mode add more special indentation words and keywords
fix remove log
fix app uni uploadfile的header属性 支持非string类型
can use same matrix
swap queue order
update pipelinemanager js
partial rollback on config factorization
test update basic dev components module
be explicit with element closest s return value
missing call keyword
scheme mode more indent fixes
fix app 修复安卓调用 getscreenbrightness setscreenbrightness 会触发 onshow 的问题
make docs crystal clear
update basecamera js
update rope js
update container js
fix async confirm add checkboxchecked to callback parameters
build remove console log types
build svelte include svelte files into package
update spritewebglrenderer js
remove unnecessary line
update rendertexture js
fix bug where the isoptiondisabled builtin was incorrectly called
optimized updateslides code
restore the changes to the slidesformargin
update webglrenderer js
remove unused code
re work to avoid if else per pr feedback
exported memberscsvimporter from members service no issue this will allow the controller to be updated to use the new importer
add fallback behaviour
remove useless whitespace
fix for when the br tag is used
test link target
link or click must both work
increase server stop timeout
restore memoization of getcomponents
memoize buildfocusableoptions as well
better method name reset once done
js fix printd issue with negative number
fix standalone fix upgrade command
send dimensions and reset scissors
fix fix stylus config
clear just the color buffer
update the projection matrix on bind
add getaspectratio method
update basecamera js
should rebind shaders too
pass the container to willrender
enforce this on container children only
fix calculate text dimensions after wrapping message text
fix exclude text with line breaks when parsing wrap setting
merge branch dev into alpha
remove follow from render code
refactor normalize module path
refactor normalize module path
fix app 修复 cover view 样式动态变更未生效的问题
add test coverage thresholds
fix auto completion for epicker in firefox
data type fix
fix missing return
fix aws lambda improve image property validation
fix app 取消 ios 真机 canvasputimagedata 压缩
fix linting spaces and semi colons
fix virtual list relative size
fix update zh locale meridiem noon
rename preact proxy to preact local proxy
remove another let in an addon
test packaging cleanup
test packaging fix plugins shape in fixture
feat weex add text u text to unitary tag
fix more types
ci ensure to update tag only after succesful binary uploads
ci temporary patch to handle china upload issues as non fatal
ci correctly log error when china upload fails
remove timeout in annotation integration test
chore ignore distro deps release
refactor consistent behavior for resolvemodule always prefer explicit paths config avoid implicitly depending on pwd
add this types to component js
fix scroller模拟滚动 移除使用event字段
update fix match media bug
fix app uni previewimage 保存失败增加提示
remove a type of post pipeline instances not only a single instance
remove references to expressions and image preloader to improve memory collection
fix shadow not complete
fix import jsonrpcerror
fix app manifest 支持全屏配置
update index js
fix potentially missing context in logger for popup entries reported internally
fix fix ukrainian uk locale typo
fix tray worker cannot find module lodash
fix unicloud db 组件在微信小程序上出错的问题
add prevtype to constructor
prefer dot syntax in test
javascript mode make sure type props don t consume closing braces
add appendchild shim to portal
fix injection test
error when mounting duplicate plugins
fix app v for recursion
feat cli ignored md
eslint config base add disabled no unsafe optional chaining rule
feat app uni saveimagetophotosalbum 支持返回保存路径
fix split indirect refer js module into subpackage when split chunks
drop container reference on portal unmount
merge branch master of https github com photonstorm phaser
docs explanation why deployfunction doesn t have to resolve images
test config schema make test case less ambigous
fix error when spying on console
karma show flat console logs in a single line
karma strip log prefixes
karma don t wrap top level strings with quote this is the same behaviour as in node
karma improve stack trace formatting
karma fix wrong filename in stack trace
