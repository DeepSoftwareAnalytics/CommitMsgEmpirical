made message tmodel ctor public
remove unused method
add method to dynamicdictionaryvalue to handle default values
merge remote tracking branch upstream master into fix song select context menus
fix merge conflict
add domain property to url
side flashes should not be affected by parallax
fix repairable not accounting for multiple icallfortransport traits
fix animation might run twice
fix resovle codacy issue
fix typo should descent
fix gameplay cursor showing in editor
allow defaults to be null when constructiing templatematcher
update title bar on connection change
fix possible nullrefs
fix test runs under headless
fix social browser calling game wide load stalls
fix follow circle being scaled far larger than it should be
add premultiplyalpha as parameter to compression
add a test for a bug
preserve parent namespace when serializing lists
chore macos ensure itemsstackpanellayout is initialized
chore fixes for buffer bytes writing
remove spurious extra fieldloader load call in map cs
remove flicker when external tool toolbar entries are changed
remove whitespace changes
htmlstring add fromfile static helper method
fix some doc comments
remove unnecessary space before parameter
get one insted of zero previous messages
prevent a crash in autotarget hasvalidtargetpriority
remove unused text transform helpers
add icallbackpublisher to di
add the public key to the client tests
don t offer to add a null check for an unknown type when one already exists
disable upnp on local games
fix semi colon issues preventing build
forgot gh action is setup to run as release
update releasehandle to check return values
made a ui buttons text translatable
update iwebbrowser xml doco for get set zoomlevel
fix bug in visitstloc do not copy null literal pseudo type
remoting increment the context id counter atomically
system core fix ice for older library lacking contravariance conflicts mcs class system core system linq parallel querynodes querygroupbynode cs
add unique constraints on rulesetinfo table to ensure things stay sane
switch back to task run
clear session error messages after successful type inference
delete useless exception handling
make withproductionoverlay fields readonly
remove annoying debug
disable sig help in cloud scenario
fix xamlvalueconverter sometimes convertertype exists but lacks default constructor
allow authorize endpoint as valid return url from user interaction service for custom redirects
add license disclaimer
testing to see if dotliquid works on mono now
fix encoding issue
expanded traceconfiguration test
fix incorrect beatmap usage
allow forceattack to be allowed banned on unitordertargeter
fix defaultdirectorybrowserinfo naming
group readonly fields
fix issue where secdataconverter would fatally fail
moved editorinfobarservice to its own folder export as workspace service
remove impossible null case dummyworkingbeatmap
changed behaviour of refresh button in setupscreen
rename maximumcombo to highestcombo
fix a crash with null textfields
fix pcl build error
remove unnecessary if debug code
remove level param for nvenc because it fails to encode with it
task slight tweak of waitany unit test
escape regex chars in script blocks
pass cancellationtoken in jobject and jconstructor writetoasync missed by mistake
add base for safety
skip a flaky test
linker do not remove resources from mscorlib unless it s action is set to link
fix some broken tests
fix non compiling tests
remove dead code
wpf remove disable gpu compositing
cli npm update directly set preview as version
chore xamlroot simplify logic for root retrieval
remove inadvertant test code
call gc after edit code dlg box closes
adjust base value
fix compiler error
fix broken vis of units in shellmap
disable test on mono managed http
update abpcarouseltaghelperservice cs add div for carausel items
fix compile errors on osx
fix ai harvester not resuming after chronoshift
reference whether texture is available rather than disposed
adds registerindicator timespan overload
converted startup tasks to a list before enumerating
remove unused method
fix the build
include white background bitmap in clipboard
fix timeout and useragent not being set in async requests
fix crashing bug in sample view
register ihashprovider before configure so it could be resolved from ioc
fix metro build
remove unused code
and another one to retry
fix broken build by add missing namespaces in protobufserviceclient
update jellyfin api controllers librarycontroller cs
changed the message ctor to remove the datetime requirement it will also be datetime now
update applicationhost cs
fix alt on gtk
fix wrong condition
fix macos crash when finishing animation
merge branch dev
tidy up skinnablecombocounter class slightly
fix introduceunsafemodifier visitstackallocexpression size expression might contain pointer type or resolve result might be null
ignore classes that are not entities
updated tests for nancymodulelocator
wrong constructor initialization of refund query request
adds unit test
remove redundant code
allow blobs in object src csp
fix out of bounds exception during indexing
spelling debug src system management automation singleshell
fix cast error
fix typo in rescanmovie log statement
owin set httplistener ignorewriteexceptions true
add a note about enemywatcher being required for on player discovered
fix port sort
remove default argument usage
fix unit test failure from merge
fix pushover priority values
merge branch master of https github com servicestack servicestack
add deserialization test and cleaned up the tests
use new internal subscribe shortcut for tasks
update restoredefaultvaluebutton when default value changes
fix name of method in release
update src features core portable converttointerpolatedstring convertregularstringtointerpolatedstringrefactoringprovider cs
remove ensureexecutable method
rename variables and make public for now
fix diagnostic ids
add implicit string converters for keytime and duration those converters are used by the generated code
fix code styling
add missing toollocationhelper methods that will be used by refsrc system web compliation assemblyresolver
move anchor and origin so metadata doesn t move if it s partially empty
update iabpmodule cs
update userlookupservice cs
fix for differences in non winrt systems
changed default engine from aesfastengine to aesengine
add nullable annotation
add constructors to identitydbcontext to allow dependency injection the other approach overriding onconfiguring does not work well with unit testing
fix log and replay file paths to save into user home dir
dx make soundeffect platforminitialize private
bring hls segments back into apple spec
remove template file type
add none mysteryman and retro as styles to the fallbackservice enum
lower default limit of conc connections
fix reflection usage for net core
merge branch master into bp iptc
add issue numbers
remove redundant string interpolation to fix ci
plinq turn wrap into an extension method
makred url class as sealed
powershellrunner list the path name that failed to hydrate
updated get epg
cleantitle remove brackets
update abpentityframeworkcoremodule cs
ensure game is at main menu before performing exit on screen
fix compilation on ios
fix copy pasta
chore styles wasm fix sample compile error
only draw frame if begindraw returns true
argument validation cleanup
expand method summary
cleaner way of disabling tap after double tap
revert https stuff
update baseplugin cs
named control add to namescope tests
add comment about language version used during decompilation
fix a null reference exception
use stable update
fix duplicated namespace
allow writing to fields even if initonly is true
sl core fix stylecop issues
what if there is no vendor info to remove from version string
don t set a default serverselection stratergy in case there was a fault getting the servertype
update follower count api source
fix spelling error in local variable name
update people fetching
bug fix day after market holiday check if sr is null from previous day
show hand cursor when hovering links
merge ilranges of delegateconstruction newobj ldftn into ilfunction
give up for now
update maincontroller startmode
use current instead of bindable bindable is going to be remove upstream
moved testcasepauseoverlay s pauseoverlay action setting to the initialiser
run check for unused fields only when compilation succeeded
merge branch master into box shadow api fixes
fix implement abstract class test
revert unnecessary file changes
only use gettypefromprogid on windows
keep to int
fix beatmap removal
now that we check actors and players we can t exclude them
add symbol to quotebar reader error logging
add virtual to tenantcreatedto
fix steam build
enable redisservereventstests in ci tests
add a null check to prevent nre when playing the no video version of a beatmap
add forward and back to the public interface
more global state hacking
fix setting count for recent scores is overcomplicated
add implementation of containertype property
fix song select wedge sometimes not updating when localisation changes
fix a crash in updatemapcommand
resolve web client playback issue
manually manage the name scope for control leak tests
restore one test
update qsv encoding
re enable paste as new button
monkeydoc fix generated ecma url format
remove trailing whitespace
remove allocations due to string format in getfullname method
perfomance optimization we should not parse product attributes twice when a customer chooses new attributes
add x forward for proto support
made loadhtml register a one time use handler
remove some unintentional whitespace
remove unnecessary override
reflect pr feedback
optionsservices getdefaultservices call no longer required this is in response to a recent change
merge branch master of github com mono mono
filetarget getarchivedateformatstring fix sonar lint warning
remove our data source when we are disposed so that far is cleaned up immediately
null ref fix
remove the ignore data member attribute on people
add extra enumeration large size to bootstrap modal
use new arraybuilder pattern
system fixes crash during implicit file uri parsing with unicode characters
torrentleech correct mst
restore security marginmodel property this property has been restored and is now an alias for security buyingpowermodel
display tmp files in the asset browser
add async moniker to getimagestream
fix an error with looping
fix for failing test case incorrectly used compare method
throw endpointnotfoundexception when it cannot connect to the remote server
update src features core portable externalaccess pythia api pythiasymbolsorting cs
unskip an iargument test
remove unnecessary index resets
add basic xmldoc to hitobjectcomposer
fix overlined playlist test scene not working
decrease the split string threshold in dashboard
made runmode obsolete default to series
make test less aggressive
cleaned up code and typos
blame spinner not dismissed for empty files and errors
few changes according to code review
allow to customize which terrain types are considered water by withcratebody
update detection of compressor
problems with missing options in earlier config
do not keep the wrong password in the textfield for next try
remove read ip from header
increase infantry crushability slightly
servicecollection fallbackservices no longer exists pass the fallback service provider to buildserviceprovider instead
no response for device put token api
upgrade rule for rof reloaddelay renaming
remove redundant bindtarget usage
features featureconfiguration cs documentation changes
update pf sd with new hit results
put list values in their own row
resettextandimage in commitinfo before loading the new data otherwise there are strange issues in blame view
modify textblocktests namespace for ios whitelisting
animetosho remove redundant from url
cache doesn t need clearing when resolving uncached feature
add private ctor in fileappendercache class separate the constructor of the empty fileappendercache instance for clarity and document the class
controlhotkeys check for designmode in onload
normalise urls used in jsonhttpclient resultsfilter
react to ef namespace change
don t spew out usage if the quiet flag was passed
make some fields readonly
do not crash in blockcontainer checkinvariant
remove elasticity from dialog appearing
add xml comment for lettercasing
remove unused done
clean up hitevents after use to avoid near permanent memory retention
larger fallback url for mobile
fix notes not getting accent colours
fix artist being the opposite value
use metadatareader to extract analyzer display names
move serverstarted notifications into the server thread
merge branch master into fix legacy skin text
fix invalid cref
fix bmp dimension check
fix non ibeatmap usage
missed a word
add ignoreextraproperties extension method for automapper mapping
changes per notnull pr comments
add streams property
add virtualfile and virtualdirectory ctor s for staticfilehandler
remove inline comment per style feedback
rerun bound node generator
fix a race use different event to fresh build error list
add back song select wait
don t auto save only save when a sync point is created
merge branch master into dont retry on replay results
add missing new
swap tab button with alt for resize location
should set language cookie path to applicationpath
resolve merge conflicts
change range attribute typeof double to decimal to match price type
minor refactoring in tag helpers
fade volume back up on pop out
lib change default install location in defines
add name to autoqueryviewer attribute
add config setting
cache tostring state in type
log error when selfhostdeployer is unable to start the process
remove duplicate events
only back out if none of the nodes match
link to relevant issue in comment
fixes semicolon in view property
fix bindable potentially being set from background thread
improved episode image discovery
update moduletemplatebase cs
fix glitch from my dumb hack
fixes maps not being able to force tech levels
use patterns in more places
comments on chart subscription
fix osu logo potentially appearing during game load in a weird place
remove website prefixes with dashes in url
don t use a string ref to the assembly
fix potential nre in statementbuilder trycatch
remove unnecessary changes
add cell check to radarwidget
fix permission attr
chore adjust x bind resources component generation name for readability
fix fallback to default lang when missing translations
temporarily remove smalltickhit tests
git version check was remove from settings
increment actor generation on cloak
reacting to hosting changes
remove double registration of icreatescommandbinding
fix storyboard video playback when not starting at beginning of beatmap
squash some spurious warnings
fix sia and trailing regex
don t set the mainwindow title then since it just makes things confusing
fix the mod being not resetting
remove unnecessary field storage
fix new fix cop error in core
dont really need to setup palettes every frame in resourcelayer
name and export option properly
morethantv improve search results
use per tile color in minimap
add typeforward to the content type names
fix missing tag from liquidate call orders that are placed with liquidate method did not come with the tag
return empty byte array instead of null
create default options in sqlserverstorage sqlconnection ctor
remove internalaccess to azureutils from orleansruntime
fix cloak modifying the palette of decoration renderables
fix wrongly placed setcanceled call
fix type in httpglobalexceptionfilter changed meesage to message
add support for passing in characters in route parameters
add iresponse endwith ext method
add tests for parameter lists
improve rendering when resizing
updated documentation to indicate required elements
keys keep keys in ssh subfolder
corrected order of assert arguments
remove unnecessary old code
update in line with combo colour api changes
change default value of border size and rounded rectangle radius
allocate a smaller apphost byte
implement a couple of socketasynceventargs methods
increase search result limit
remove unused using
reduce x86 release baseline to try and make up for flakiness
display missing pr text in the comment tab
save all the files
fix apimember name with apiallowablevalues
move from getcacheddiagnosticsasync to getdiagnosticsasync
assert on ui thread
remove strange using
don t disable everyone s shroud when someone s gps goes up
change default mouse actions
allow passing in the body content as a string
handle no subs setting
more code clean up
add these to mobile as well
remove string literals in registry provider
toolbar buttons should handle on click not mouse down
make selectedhitobjects into a hashset for quick contains checks
gazelletracker add support for imdb search
more debug output from cg renderer setup
add symbolkey test
disallow clicking through the song select footer
don t limit keyboard input based on menu button scale
add missing keyword change
fix complier warning
log app dir
fix the testfieldsforequalsandgethashcode test
remove unused empty methods
fix error getting account option holdings from brokerage
changing to list
winforms controlextensions invokeonuithreadifrequired perform no action if control disposed ishandlecreated false
add rest of enum descriptions
remove not useful comment
fix syntax warning for playfield children
remove test code
chore add interface
use colour for non faded scores
fix spelling error trancated truncated and reformatted call to string format
remove unused usings
remove testing wait
renumber errorcode to avoid conflict with nullable prototype
add a desc for buildable buildpaletteorder also removes the long outdated ui fluff comment
these tests can be run now
gitstatistics plugin compilation fix
fix ios date picker not closing
fix spinner easing with hidden
explicitly document backtofront and fronttoback use an unstable sort
fix a few compilation errors related to the previous merge
update test case
fix ioperation cfg test hook
listener fall back to polling if irc connection is dropped
uncomment get check for url redirects
fix enumeration values
dynamically position the timer so that it isn t cut off
forceinitialize with propagateargument
unroll loop for the last control point
controller update comments
fix nre for unknown method token for some generated methods
check for entertunnel frozenactor actor is null
add missing captures list on android and macos
allow in search
fix problem with the duplicate root directory while creating an absolute path
wcf fix index of message body parts do not use messagebodymemberattribute order as index
add configurationprofile paramter to electronize start command
remove unused parameters
build break fix remove missed new constrain which is not required anymore
make method async
myamity fix login detection
fix commitbutton bottom border
escape characters in xml documentation
fix wrong parameter for token
mouse selection don t select a hidden line to prevent scrolling
fix cargo loading
change editor timeline mouse wheel handling to scroll by default and zoom with alt held
remove usage of case when caught me off gaurd
add ib brokerage message to invalid order event
add imgrush com support
remove not used localizer field
set missing widgetdefinition properties
add required types to testcase
remove redundant cast
blue tigers fix category url parameter blue tigers fix category url parameter
fix taiko difficulty calculator never considering mods
correct copy error need to check that complex cues exist
add missing requires to missionobjectiveproperties
remove old actions
use antialiasing filter when rendering world annotations
move line inside if
fix assignment of reason
remove asserts for now
update licenseusagetests cs
helpers only allow one thread to write to the log at a time fixes some crashes
add blank lines for conformity
system fixes build
getallcountries cache by store
fix null reference exception when using multibinding without stringformat and converter
only overwrite values if they re not default values
merge remote branch vasily kirichenko git word replacement bug
add compiler directive
factor formatting code out of cashtick ctor
merge branch json schema of github com jamesnk newtonsoft json into json schema
close commit dialog when all changes are commited now considers new file as a change
merge branch master into gitsharp
change getlastmodified api to return non nullable datetime
fix height being set instead of width
remove warning message unused variable
use string empty in icontooltip
really fix the undeploy while damaged crash
don t disconnect every client when one user gets kicked
address test failures
add a test for serializing a single connection node
update thread pool check
include name in streamfiles error responses
implement iequatable country
changed parser registration
change cachedhttpclient to accept nocache
handlecallinstruction do not add casts to anonymous types in arguments
fix negative dates and time moving in opposite direction
update totalclassifiertests for top level statements
image size form must be top most
add missing xml comments
update nrefactory to fix incorrect method type arguments on calls to non generic methods from within generic methods
setup controller remove leftover startup item logic
fix rank status
re enable gpu in offscreen and wpf examples
fix build error
tidy subscriptiondatareader time checking
change waitasync so it s pcl compliant
forward any unhandled scroll events to volume overlay
not used in portable version
update auto organize
allow load screens to override startgame
bug fix in indexin
remove explicit type info
update notificationoption cs fixes serialisation bug
merge branch master into fix testcasebeatmapcarousel
fix incorrect sftp upload result if directory created while uploading
fix missing namespace
fix the one lame test for the default view resolver
move method above nested class
better initial beat handling
we should clear cache when some error happened during installation
fix visual settings expand button colour
mcs less cascading errors
add startup value for the slider
add unit test
merge branch jb travistests
add test coverage of animation restarting
fix fallback logic for blazor attribute localization
use correct width adjust for osu catch playfield
remove double dictionary check
merge branch master into manual position adjustments
add new pixel shader extension methods
make bot refreshcms not fatal works around latest nightly mono fuckup by the way
make fields readonly
add mocks for geometry impls
disallow sound radar notifications for neutral actors via a configurable field on the trait
fix relative position being compared to time
visualstategroup transition were not working well on first time
i am good at this
fix osname on ios
fix skia adjust error handling for skia samples run
change to use kotlin companion object to hide field from gson serializer
remove system net sockets
add missing null check
remove unneeded highfrequencymouseposition from gameplaymenuoverlay
merge branch serializeperf into serializationkinds
registered iperwebrequestdependency in controllerconventionalregistrar
fix unscaled scissor rectangle being applied after rotation
only return true on mclick
asp net fix a thinko in one of the previous commits
set zipentryname datatype to auxiliary
updated futures canonical symbol representation tested local backtests local live iqfeed and ib
add requires attackbaseinfo to attackorderpowerinfo
fix rebase error
forgot shutdown the driver at exit
add missing license header
test that closing handle will release managed resource by calling dispose
fix iswindows10upper not assigned caused nettraffic won t be displayed at the first start fix supportfakedns flag lazyload
appveyor now have innosetup installed so no need to download
add a user object for now
fix hotkey and recent persistence
fix complier warning
allow retries of ssl exceptions
remove unused field
remove duplicated dependency on audiomanager
show pii in dev mode
xbuild publicsign and delaysign cannot be used at same time
remove unused using
serialize type as string in ssref langs
remove empty lines
fix compilation on monomac target monomac lacks opengl extensions
fix for ios not calling onorientationchanged
update docs header to reflect using rsa oaep
tidy in tileset
document the meaning of the long values where timespan is no longer used
remove protocol from host
made us to re analyze some part of solution when assembly name have changed
use braces for if statements
windows fix build
untangle default selection in serverbrowserlogic
handling process end on integration tests
update hls subtitle display name
formatting and remove of comments
add user warning for expired option holding
use priority correctly in filesystem openwithexts
make toolbar icons the same width as options icons
better check for null
fix backgrounds not properly being faded in song select
merge branch release into dev
only schedule when faulted
getline fix the logic for advancing the top item
fix replays being parsed with incorrect cultures
mark a couple of tests clronly to work around flakiness
workaround monodevelop core filepath tests failing to run due to strong name issues
corrected function calls involving iterator function
fix documentation for extraruletype enum values
make sure to update image size label theme
delete dead code
fix inverted condition when authenticating with an apikey
update android and ouya activity templates to work with these changes
remove unnecessary parenthesize
fixes brokerage message in brokeragemodelalgorithm
remove using antialias for straight lines to avoid blur
use onlogerror for uncaught service error logging
fix broken soundeffectinstance islooped on android
add doc comments to isyntaxkindsservice
disable flaky tests
fix resolution problems on android when resuming from portrait to a landscape game
changed the default document
bugfix to prevent masking errors with a transaction error
apply rounding to bar line start times
add test of graininterfaceutils computeinterfaceid
update accelerometer android cs
names for tuple returned by buildwithnamescope
do not emit pdb path in pe file if not generating pdb
loglevel can be set to any value except off for this method to create a directory
remove defunct iordercursor interface
fix merge issue
revert remove creategroup order as the actorgroupproxy is gone
resolving merge issues
use refereence value
set isstarted after the listener has started
created helper method timezoneinfo isindst
don t cache country response
sighlp use normal content
remove test code
remove schedule causing default skin explosion regression
bcl add workaround for csc bug
fix system threading tasks dataflow version info
made fix all test to run on closed files
add stringextensions concat to fix microsoft io redist assembly signed off by dotnet bot dotnet bot microsoft com
expose the id of the service process launched from winsw can use these information to talk back to winsw exe
merge branch master into spriteicon invalidation fix
filehistory reorder tabs as in browse w diff before view
add shadows to toolbar tooltips
set default accentcolour for hit objects
fix imported map spawn ownership
enhancement for tests
apply review suggestion and fix entity concurrency tokens
trying out fixes
unwrap response dto
remove the validation check from the csi task as we only allow a single source file specified
fix test mismerge
add optional prune boolean parameter to gitmodule fetchcmd with default of false
fix nil handle bugs in basetypesentrynode
system add acceptwebsocketasync signatures
fix spacing of the status window
move mathex to uno package
unhardcode map installation directory
hide revision graph when any kind of filter is active
add missing copyright header
windows eventlog fix font
update soundeffect cs
cf doesn t support the content of appdomainhelper add compiler directive to exclude content
force case senitivity to be false
jwt already being deleted
fix namespaces in validationresultextension
don t default classifications if colors have been customized
add waitforexit for mac issues
ignore test cases for ci
change clientcanswaptemplatesattribute to reqfilter
include size of binary file in message
disable scrolling speed control for osu catch
danishbits add blandet category
remove debug code
remove unused usings
add function description
pass variable explictly
fix test method to use the same time
update xml doc comments
disable cas permission tests for mobile profile
null fix for live trading backtesting brokerage
add custom xaml namespace
use news overlay resources
mremoteng import doesnt need to be static
update xml comment
add file lock logging message
mediaplayer seek on android
fix syntax error
remove unused property getter
remove redundant code
writeobject t needs to write type identification for value types to match xna behavior
trim and tolower billing email
add livenodepacket disableacknowledgement allows a live job to declare that the job shouldn t be acknowledged
revert revises gzipstream ctor parameter names
add graphicsadapter iswidescreen documentation
slight tidy in infiltrateforsupportpower
cascade potential facebook errors
configfiletest mono support
fix memory leak caused by shared plugin instances
fix build issue
don t allocate anonymous method container when importing event fields
remove unused grid test variables
fix padding when user has no additional info
implement humanize test localization for current culture
replace with local tolist
restore bool char array ldelem stelem instructions
xbuild fix installedassemblytables signature
rename method parameter
use minimum width on statistics columns
logassemblyversion embedded assembly location can be null or empty
react to logging changes
merge branch loglines into simplifyalias
replace the float waterchance by an integer percentage
no need to send userid to permissionchecker for current user
add nrt support
shutdown in viewer mode on exception
fix startswith snafu
set playercount based on waypoints
move file check
fix none result getting included
add getformdata html helper
selectel cloud storage add https selectel com services cloud storage
hook up the get from user button in the github search for repositories dialog
remove unused method
add patch to default cors methods
add further xmldoc
update resizehelper cs
fix offset when copying to buffer
controller remove puctuation in notification header
adjust suggestion tag clipping to prevent seeing errant pixels drawn on the screen
fix quarter month bug
skip raft tests but merging all the other hardening i did in that branch
fix remove wrong increment
use int for total hits in osuperformancecalculator
extend using inotifybuildcomplete to production
fix hold note judgements displaying incorrectly
don t offer to convert to pattern matching when type is nullable
disable editor vertical ruler by default
add messages for zero unit price reasons in cashbuyingpowermodel
update recording deletion
don t trigger request if there are no avaliable users
avoid errors being logged when importing beatmaps while logged out
fix bug in checkforchanges
flyouts detect theme fallback main window maybe for later look for application wide theme
fix more post merge errors
read tempdir command line option
missing null check on deferredtaskengine
nothing to see here move along
log to console is not quiet
add cef initialize check in constructor of winforms and offscreen implementations makes them consistent with the wpf version
undo test fix
update the code
only apply series image aspect ratio if episode season has no primary image
resolved bug where hitting the delete key would bring up the prompt to delete root nodes and putty nodes
ensure there is enough time before the first object in osu roughly following osu stable specs
modify class access level
fixup pattern we look for before offering
reset module name when reloading from byte
renamed static to mapstaticcontent
check that the parameter is null
remove dedundant this qualifier
fix nre in metadataassembly
remove usage of typeswitch
actually make into a struct
add debug writeline to send calls
remove the element if setelement sets the value to null
fix test for spelling
fix comment text
apparently people do not understand this part clear enough
update schedules direct timeouts
copying the aspnet loader dll only on non kpm bundle scenario
add dbgprocess pointersize
updated commands layout
update src features core portable usesystemhashcode analyzer cs co authored by jason malinowski jason jason m com
add extra test
add null check
add requestinfo as normal plugin but remove after configure if debugmode
avatars don t download further if there s no internet connection
remove stray character from description
lazty hostenvironmentinfo osversion
add missing fonts section
made the gcs backend available again
unskip tests we can handle now
avoid duplicate work previous code would compute statistics twice for no reason
remove reset functions
apply message response filters in mq requests
flattened inheritance chain
fix release build
stream copyto is nice to have everywhere
simplified math add comment
remove unuse import packages
cleaning up a warning
remove fireoncurrentvaluechanged and ios calls since it appears to be the only platform still using the method
fixes some comments in composer cs
react to merge conflicts
make resolution of dataqueuehandler overrideable
unwrap codestyleoption t without allocating
add gameplay screen specific video
disable ftp tests
fix forced seeding in qbittorrent status treated as complete
update ditherextensions cs
one small change to quandl subscriptiondatasource
same for double
react to http namespace changes
nuke physicalconnection onheartbeat unused
made startup a singleton
fix bad using in last commit
fix succeeded failed job counters on history chart
listen to group commands only on master chat
fix ci issue
fix typo vs in generateuniquename
further tier price changes
fix contentidentity of texture2dcontent is not set
fix mediacoversupdatedevent broadcast cleans exception when updating media covers expects a series id
fix mosueup and highresolution events not being handled by menuoverlays
improve status trace to include status message if available
revert changes to screntestscene
default smtp ports
windowsbase fix incorrect test
update primary roslyn processes for test diagnostics
fix broken test
add version number to the common project
inform the user which sound engine is used
infra ifilesystem get_current_directory
remove unused argument in string formatting
semi revert removal of name for similar in openapi
setup change some wording
open browser in vs rather than externally
fix build errors o o
add resize width and resize height
write setup power plan in a separate line
fix translationapp build after changing pluginregistry initialize method to synchronous
fix cron monthinterval method to return correct expression
fix rollingcounter not updating initial value if changed before loaded
fix custom palettes in actor previews
update src compilers core portable diagnosticanalyzer analysisstate peranalyzerstate cs co authored by sam harwell sam tunnelvisionlabs com
update mediabrowser providers tv missingepisodeprovider cs co authored by cvium cvium users noreply github com
new repacks will be treated the same as propers
remove duplicate configurationresourcemanager use appsettings instead
update featuremanagementmodal razor cs
use patterns in more places
return the most specific error handler first
add a dumb way to get the default resolver
made epochtime readonly
make sure it only loops for themed intros if true
fix processmessage issue
merge branch release into dev
catch exception when trying to kill polipo
motd failure avoidance
made video size richer
ensure changes are ordered and non overlapping
repo fix the getcommits regex to also parse gmt n timezones
don t clear selection if right clicking inside it
replaced inconsistent task fromresult usage with task completedtask
add some documentation for workspace setdocumentcontext
stop writing the messageid twice on sse
expandoobject is a dictionary and should be mapped as such
fix attribute spacing
add null check in orderticket cancel
system xamarin_start_wwan is not available for tvos and watchos
remove culture verification method as no longer used
bypass fail on replays
handle input again
improve messages when tests fail due to expected text
fix crash on surrender
validate resource key before using formatwith extension
make toolbar button abstract
winforms cope with null hwnd value from hwnd objectfromhandle
menuflyoutitems can now receive datacontext
disable basic ee integration tests
add documentation explaining intertwining logic
update volatilityetfuniverse cs removes dust which is an etf tracking gold mining activity
ref returning an instance event from a struct should be an error
put a slight delay on directory watchers removetempignore
add newly add types
silence exception when staging unstaging lines when only lines in the header is selected
fix the mobile build
apply suggestions from code review co authored by cyrusnajmabadi cyrus najmabadi gmail com
modified validator not found exception message to be more generic
limit calculator trigger must has one number at least make not affect some keyword like pi
mono debugger soft re add dispose method on virtualmachine for api compat
check if url is a directory
uncommented examples for intial cloners
revert unintentional change
handle when pathinfo is empty string
update accountmanage razor cs
standardized is default for now
expose setting clipboard text to mod code
use javascript for nested settings design by default
remove pixel format property
ffmpeg internal aac encoder is stable and recommended now
remove unnecessary setters
add show tips to options menu
add test for patch appendtextline
account for headless nulls but how
surface uinavigationbar ios constructors
make multipliercontrolpoint s starttime variable
merge branch feature cps of https github com davkean roslyn git
android s guide showkeyboardinput now uses usepasswordmode parameter
log sound parsing errors in sound log
write empty rootnamespace if need
remove debug message
minor formatting change
remove redundant access permission
instantiate auditloginfo comments in the ctor
return module modelvalidationresult
remove unnecessary todo
remove usings for namespaces which no longer exist or are not used
move out usings
adjust icon scale of play button
update with pattern changes
fix framestabilitycontainer performing frame stable seeks
remove unused linkproc function
core automapper now can stand alone
reduce logfile name
add missing semi colon in test
fix null reference exception in guide when called from threadpool threads
fix cloud sync playback
add nancycontext to owin environment
fix possible bug in dotnetcorepathfinder
fix spinner crashing on rewind
prevent throwing exception when cast fails
ensure audio is pre prepared for lower latency playing
more robust replay file creation
ssh auth clean up
increase duration between spawn for sp
return the correct length for input streams
fix app menu placement
remove unnecessary usings from microsoft framework localization tests
minor ishippingservice changes use ilist t instead of list t
treeview navigation never return handled if returning null
escape shell executable path in commandline
rename f grade to d
return an empty task from reconnect in tests
tidy in shpimagewidget
remove obsolete property i m not gonna go semver on this one it s been obsolete for a while
address review update log message
run onselectionchanged on each change
asp net ignore javascript blocks enclosed in html comments
remove usage of callcontextlocator
update xsd generator for timespan iset hashset ilist ienumerable
return early if type of validating object is primitive
minor style changes
ensure we default t in case of a struct
fix test case so it works cross plat this just happened to be working on windows newlines
mono symbolicate load assemblies in reflection only context
remove unused namespace
null model check
ensure a playing song is stopped when switching songs
don t move completed drum rolls above the playfield
add index to cacheentry expiry as it s used to delete expired entries
make sure we don t get stuck in an infinite loop
update next up query
don t calculate performance if difficulty attributes aren t locally computable
route disconnect back to game
fixes nullpointerexception when frozen actor is null
reacting to verbose rename
set openapi schema type to file where possible
remove exchange check
add missing newline to xaml output
window position for windowsgl platform
linux fs is case sensitive
add sumary and remark to class header
add bool check as well
enable use of aliasas for multipart item name
use trygetvalue when getting remoteendpointmessageproperty
disable blame menu items if revision is not in the revision grid
iptorrents update alternative site links
disable failing test
add a tween when clicking to seek
deprecate tomimemessage extension
ffs i m a tard
add test getlistasync
improve the applicability span
hacky fix for map upgrading
disable jenkins logging until needed again
fix status control position on multi monitor system
differentiate new old lua apis
initialsize option is not supported for encrypted datafiles
minor changes to comments
use async apis internally
make slider circle masks not handle mouse input at all
set initial dailyportfoliovalue for livetrading
annotate and seal doubleclickrevisioneventargs
made preparecategorysimplemodels method public
add func for calculating scaled width
remove exceptioncandidate non transparant and thus removal
add failing tests for point parsing
add setting to start fully wounded
add response syntax to custom uploader
tell users they shouldn t use basemetrodialog
update params when stream copying
don t create a new request if onlineid is null
minor rewording missed from pr
torrentleech fix download link again
fix keycontainerpermission setflags
copy paste fix
don t reset the synchronization context
modify severity of fails those are now client sided errors so they re serious yeah like everything in steam
better error when no sequences defined at all
fix typo fixes a typo in the summary of syntaxgenerator throwstatement
skip authentication middleware when database is not installed otherwise an exception could be thrown
handle ctrl c while prompting
ensure prime quotes are html encoded
fix avaloniaformattedtextline line width
fix linespacing to use the font metrics
xbuild update printed version copyright
made weatherclient disposable
make methods synchronous
remove extraneous interface add
misc archibot optimization
add assert isnull quotebar check
update smart match tab
describe versioning statagy
add some comments
fix codefactor issues
refactor improve test name
moved suppress finalize to the right place
fix pointerover on mouse leaving window
moved event firing to after savechanges
set default normalization as adjusted
add link to the tracking issue
plinq remove the scwl noise in parallelenumerable testorderbyageascendingthebynamedescending test
only apply taiko legacytiminginfo to legacybeatmaps
select the expr in the watch window when editing it
remove test code
udpclient receive often returns null not sure this undocumented behavior is correct but workaround anyways
add f to languagesettingspersister map
remove mods from constructor
remove unused usings
comments clean up
statusicon remove another reference to unneeded var
handle gracefully a nre on body message
adjust test to make it culture neutral
also disable selectively this test at runtime
monkeydoc disable now unneeded hack
reacting to korebuild changes
fix volumecontrol and onscreendisplay being add to the wrong container
add support for html redirect
expose editorclock for consumption
add comment to hostingfixture to clarify that the rest requires administrative priveledges or it will fail
controller use a more personal file name for the pubkey
remove misplaced usings
add missing licence headers
tweak to followup underlying pattern compiler uses
fix room name being coloured
temporary fix for older clients
load maps from support dir maps mod too
issue fix checkout attribute text prompt should be displayed localized on the shopping cart page
change existing test as error moved to later phase
more easily flatten methods
system core cache array comparison predicate
add displayname in credentials authenticateresponse
add file exit to editor menu
fix nullref with null beatmap
use a shorter type name without generic args
cleaned up iuseridentity cs
updated source code comment
merge branch master into remove include
disable the remaining failing tests for now
update namespace of gif tests
react to removal of addscopedinstance
fix an issue after the merge conflict
add two factor provider to response and device update on old auth bearer conversion
throw invalidoperationexception when attachment extraction exits abnormally or doesn t produce output
use thread currentculture instead of system
updated some source code comments of basecachemanager
allow changing the default dependency resolver for the hubs route
add ijavascriptobjectrepository unregisterall unregisterall all bound objects does not unbind them from the js context
changed build palette hotkeys cannot be triggered anymore when used together with a modifier
update tests imagesharp tests processing transforms affinetransformtests cs
add verify methods
ms build load importgroup in root
print a message after unpacking a native file
fix validation propogation on direct properties
fix skia text selection
updated scaleoutmapping so that the dictionary is passed in
hook up collection filter
add test for system io memorymappedfiles memorymappedviewaccessor readarray
apply formbrowse filter textbox filter on enter
update button colours
fix route value
update abpdbcontext cs
delegating to existing method
myanonamouse improve re login detection
reset config after updating current dir
deprecate verifycachedsessionid and remove internal savesessioninitems
correct naming of addcursorrectcursor to addcursorrect as it has been renamed in mono
follow up on merge from tuples to future
add improve xmldoc
remove unused urlhelperextension
add null checks on session device creation
remove usage of typeswitch
fix wrong attributeusageattribute
undo cancelatperiodend on reinstate
add comment explaining getminbitrate
maint corrected whitespace
don t include streamext references to gzip deflate either
add new pocolookup and pocolookup complex properties to test complex generic defintions
use correct type name
bug fix log ruleindex as integer not string
make some methods static
wpf improve chromiumwebbrowser dispose renderhandler wasn t being disposed at all used newer syntax for null dispose calls
fix c remove unnecessary cast for boxing conversions in conditional expressions
react to options
merge branch master of https github com jackett jackett
increased the amount and time logs are kept in memory
remove unnecessary cancellation
update android projs
update src avalonia controls notifications inotification cs improve xml comments co authored by steven kirk grokys users noreply github com
taiko drumroll ticks should just play the playfield samples
fix a small issue with async patch support and the synchronizationcontext changes
remove using that was no longer being used
delete unused endindex variable
update collectionviewuitests cs
allow deserialization in jsonhttpclient
mark autogeneration tests as headless
system test uri constructor with escaped user info chars
match default is null
fix post rebase errors
update dlna server name
fix remaining test case issues
more typo fixes
remove parenttag requirement from nop panel tag helper
remove unnecessary usings
update cached tradier orders on each fill
apply suggestions from code review co authored by dkanada dkanada users noreply github com
add license headers
remove the beta tag
use field for getobjectdata and this instead of base for ctor in securityexception to help linker remove more code
setup remove some strings
don t create optionkeys with language names if the option isn t per language
ensure the previous leaderboard is remove
add note that while we could implement dynamic invocations we chose not to
add information about debug info source to ilspy ui
do nothing in kestrelthread stop if libuv fails to load this prevents dllnotfoundexceptions from being masked by nullreferenceexceptions
update linux ib gateway process check
xmlformatterdeserializer handle reflectiontypeloadexceptions better
merge branch master into lightnessprocessor
update live tv osd
check before computation
fix for failing mspec test
need fix setdirty order
open url in new window browser
small changes per review
update src features csharp portable makelocalfunctionstatic makelocalfunctionstaticcodefixhelper cs co authored by jared parsons jaredpparsons gmail com
update abpdictionarybasedstringlocalizer cs
ignore coreeventid lazyloadondisposedcontextwarning
use block body for constructor to fix remaining code inspection issue
add basic sampleinfo fromsoundpoint and getchannel methods
add odd even type to test scenes
thepiratebay update proxies
remove dead code
bugfix minus port support
fix built in implementations
fix url encoding
fix a incorrect spelled word
fix null reference exception in github plugin
settingspagebase return also textbox and combobox text value to be found the the search
fix all local tests
fix updating menu item
teamcity multiple projects and filterd builds support small fix
remove obsolete source code comments
extend nullable directive
don t throw if type doesn t exist
remove redundant getdefaultuseragent
made metadatafiles internal
formdeletetag enabled autocomplete feature for tag list
merge branch master into slider controlpoint masks
mono tuner remove unused field
set log level to debug for http range requests
move absing from the apikudosuhistory
small change change comment sql server to oracle
update aggregate cs
add getallbytesasync to abpformfileextensions this makes it easier to get an array of bytes directly from an iformfile
delete unused local
move list initialisation to pattern generation code
use patterns in more places
fix encoding parameter
change mysql to mysql
fix possible error on spotlightslayout disposal
corrected spelling of comparision in server cs release mode
use actual parameter custom attribute type this fixes the situation where a parameter has more than one custom attribute
remove unused using statement
fixes test case
fix cached entity tostring wasn t updated when calling entity release
implement cr feedback
don t validate assignroles service for in proc requests
only cancel a running analysis if user has manually add remove assemblies
ignore new test classes
update flyout cs
skip over declaration parts to get to the parenting statement
fix merge conflicts
remove commented code
add more tests for imagesizehelper
fix some base classes not being abstract
more whitespace changes
pass the correct verb to validatefn
add default project imports
stop instantiating and operationcanceledexception with a cancellationtoken on windows phone
don t update rich presence if the rpc client isn t initialized
fix build error
fix test class name
add script events beforecheckout aftercheckout
remove leftover unused private methods
ilspy addin remove unused code
fix compile error
apply review fixes
update device cs
throw exception when endpointhost config is not configured properly
remove invalid param doc tag
ignore casing when authenticating users
fix conditionmanager grantcondition doc
allow wall renderers in mod code
rework the show system menu show the system menu on icon if mouse ypos titlebarheight usenonewindowstyle true
perf remove use of enumerable any in grid layout
enable mypermission group for crud page generator to add items under this group
fix warning which was reported as error
fix log format
make listboxitem focusable thus selectable
merge from dev
fix test failures
fix queue specification
test check for buffer overrun
fix repl window not always showing the top lines at startup
address pr feedback and fix a broken unit test
disable touch scrolling implementation as it doesn t work correctly
merged upstream changes
remove redundant namepace
simpler static field init
update osu game users drawables drawableavatar cs co authored by dean herbert pe ppy sh
fix failing mono test
line ending wtf
clean up comments
add internal only completionitem automationtext
remove environment variable
use exception instead of an assert
add statusdescription on webserviceexception in asyncserviceclient
fix bots not deploying their mcv as a result of the deploytransform change
original size vs target size fix
ignore extra node in baseshape
fix nasty topmost bug
updated generated class for vibrationdevice for wasm ignoring warning for unused private field
update simplequerytestbase cs
move all security headers to web vault location
missing setting disposed to true
remove unused variable
merge remote branch lukesmith browsertests into integration
doc change wording on name parameter this was done to match the wording that was applied to the source command
ensure handlers run on the ui thread
add the setting of cookiecontainer when request required authentication and a new request was created
enc disallow add an extern constructor
fix for wrong qualities showing up will be slower to load though
make sure items is defaulted to a new list
check for nulls when looking for ipc txt
lincontroller refer to html files
add in obsolete helper methods to ensure algorithms don t break with new order changes
add descriptions to the selectionsize traits
about fix background image
update pluginmanager cs changed a to i
remove usage of typeswitch
add traits attribute
redundant type args
fix settings exists check
dispose selectionboxes when they die
show generic method arguments in dashboard
add setting checkbox to toggle debug logs
typo noitification notification
use inline variables
use local input manager
migrate skip lengths
a couple more docs
better type check
use focusmanager to set focus
regenerate auto gen designer file after merge
perf initialize frameworkelement style if new parent is not null improves the performance of the deconstruction of the visual tree
add crt ext also
use a specific bootstrapper instead of using the locator
update global cs
handle surrogate pairs in gettextelementspan
fix follow points not starting at correct time
valign correctly in labelwidget
merge branch master into fix wedge nullref
update src features csharp portable useexpressionbody useexpressionbodycoderefactoringprovider cs
fix an oversight with bool initialization
move completely to fetch
return list result directly for mediaattachments
set ilranges on statements conditions
fix bug where window text wasnt updated when choosing to create new connection from dialog popup
add interfaces to apiresponse
pruned unnessecary namespace qualifier
export light theme color for string escape character
make code more readable
use startup configuration to set the default redirect path
minor issue fix in copyproduct method
update upload tool so it works
prevent emitting duplicate first points in user defined universe
add metadatafilter to autoqueryfeature
add logic to apply fix to windows only
strip prefix to prevent cookie exceptions with invalid names
fix failing tests
changed isjsoncontenttype to check for endswith json
only using optional culture parameter for public methods
quick fix for fix initializers of array of generic struct
make jamsmissiles and hovers classes public
fix duplicate test names
add null check to scheduledirect
manually set clock for storyboard if loading before being given a parent
intemplateof should only affect later selectors so return a copy of the match
put useservices extension method in microsoft aspnet
dispose of metadata
windows implementation of only draw frame if begindraw return true
visualdebugging displaying pool name in pooldebugeditor
showwindowcommandsontop for window buttons too hide window button commands when showwindowcommandsontop is false
remove unused youmaygetmorehelphere method
adjust facade scale now that the size is different
add link to pull request
currentthreadisloopthread is a property
delete test asserting for the stuff moved
fix exception messages two exception messages in entity cs were missing a whitespace
fix shouldskip inmem logger
remove done todo
remove itinial datetime kind check for todatetimeunspecified_test test
further changes on color squares attribute control type
setting servicelifetime scoped to orderingcontext so it is explicit instead of implicit by default
fix setting management tests
spec add expectedexceptionattribute override add expectedexceptionattribute override to tinyspec to keep from add the testing framework to the files
fix reference to incorrectly named object
update lambdasimplifiertests to correctly correspond to refactoring standards
remove debug noise
symboltype in reflection only profiles
some cleanup should unify mouse code further on desktop systems
linux controller set exec path to find the right subcommands
rename to showresponsestatusinmetadatapages
better group cancel conditional
make collapsing of x x y to x y work in a few more cases
automation test application current works in the ctor on all platforms
fix crashes on beatmap not being set correctly in player
move padding to fill not scroll container
remove exception this is set in speedadjustmentcontainer
fix case where the previous touchlocation was being lost due to multiple calls to asmovedstate
fix crash in crateeffect
update test efcore relational tests migrations internal migrationsmodeldiffertest cs
use relative size for ticks
add better asserts
minor temporary changes in dbcontextextensions
fix typo in restclient cookiecontainer documentation no bug left behind
fix missed string interpolation
remove unused directive
add better detection of our base dir
remove explicit call to gc sometimes hangs in mono
add ondispose sse callback
fix another test
fix the back link to return to the metadata page
make field readonly
fix pageslide direction mess up
fix case sensitivity in iskeyword check
fallbackconverter converts on to true if typeconverter is a booleanconverter and destintionistype is bool
add more detailed ib logging in handleorderstatusupdates
getentityid should read from new value then original value
order sync users
fix highestversionavailable traching in tryquerynewestversion
better test verification for pdb
remove unused usings
log startup time
mono symbolicate fix wrong directory exists check
fix incorrect exception messages
mark grainclient initialize overload for specifying single gateway as obsolete
sparkles remove todo that s already a github issue
close is successful so at least we can reelase some resources for now add a log message also
ui show even log when app icon is clicked and there are no other windows
suppress compiler warnings
add compiler warning overrides
change canextractsubtitles to true to allow conversion during direct streaming
fix reference to dummyapi not using helper property
update cancellationtokenextentions cs
add fallback fastconvert for enums this fixes adaptive trigger on splitview displaymode not working in samplechoosercontrol
missed a type change thanks for the thrown keynotfoundexception wpf
update src analyzers csharp codefixes converttypeoftonameof csharpconverttypeoftonameofcodefixprovider cs co authored by cyrusnajmabadi cyrus najmabadi gmail com
exclude nunit and ncrunch from automatic bootstrap registration
use is check
add wrapangle method patch from johnholmes
taskpool is broken in rx right now move to threadpool
react to logging changes
sorting dtos and consolidating into one function
update authfeature cs
update mediabrowser common plugins ipluginregistrar cs co authored by claus vium cvium users noreply github com
finish internal counter transformation regardless of the combo
make method private
add icon for random mod
fix scroll wheel being handled by base test scene
fix target lines still being visible after an owner change
hide game cursor so it is not included in stream
reacting to di changes
graph tooltip includes full message when available
resetwindowbounds has change signature
wrong locale resource was used for acl field on the topic details page admin area
report maxplayers to master server
fix a couple of boring bugs
polish benchmarks fix new warnings
fetcher git fix typo pointing to excludes file
comment out unused container for now
still build cache for tests
further changes on the previous commit remove unused usage directive
update onnx convert documentation limited to onnx ml target platforms
merge remote tracking branch upstream master into fix transparent background on screen scale modes
update abpstringtoenumconverter cs
update coding style
merge branch master into avoidmultiplearraylookup
add workaround for lack of stopwatch in silverlight
remove unused using statement
may as well be consistent
fix toolbar not appearing at main menu
automatically apply globalresponse headers to serverevent raw httphandlers and add onheartbeatinit callback to heartbeat handler
fix some style issues in mobile
reacting to hosting changes
don t invalidate itemsrepeater measure on children change
add previously remove return code in rangeslider cs again
remove unused var
add missing licence headers
cache before idle tracker
fix folders affecting album resolver
add property and failing test
remove some bogosity there will likely be regressions but not a huge deal
minor reshuffling recolouring
add md5 and sha256 sum methods
remove private modifier to follow the rest of javascriptserializer
fix test scene not actually covering the class it s testing
fix monitor initialization bug
set update input s id from entity before the mapping
changed custom iresourceassemblyproviders to use nancyreferencing strategy
throw exception instead of just creation of invalidoperationexception object in translationfile cs
fix validatepath response code
use a wait free fast path in asynclazy t getvalueasync and getvalue
add hangfire redis handler to mvcsample
fix build break
enable auto update
fix typo in log file
do not accept bank tokens for premium
add animation parameter
fix missing conversion case
chore add issue number to test ignore
check to make sure user actually needs key update
add default value to offscreen constrictor
revert bad change
fix on logout
remove unnecessary code
fix broken test
add missing blank line
fix some compile warnings about unused code
bugfix prefix was called backup
make getobservable return current value
display branch name in bold only when is the one checked out
plinq add dummy getenumerator override in orderedparallelquery to make gui compare happy
change scoreboard text size
unit test add
system servicepoint checkavailableforrecycling is now private
renamed paypal plugin namespace
fix for nullreferenceexceptions in authenticationinfo
releasing new version
call gtkimcontextfilterkeypress after issuing the keydown event to get the same sequence of events as windows
use any instead of count
add new cefmenucommand enum items
use shared itemhelper
apply start date filter in coarsegenerator
reduce default tooltip delay in cnc to a playable level
make ordermanager fields readonly where possible
changed the order of repair attempts to make the simple fix first
register filebufferingreadstream for disposal
fix shared comparer
merge if statements code simplification
ensure find the resources when it do not have the app xaml
implemented displaymodecollection getenumerator remove unnessasary displaymodecollection gethashcode override
enabled tick backtesting for futures
add mania statistics
fix doubletaps occurring at the start due to being close to _lasttap
add itrackingconsentfeature createconsentcookie
add docs about how the include all derived maps works
add default as an alias for otherwise
fix missing reference
maint virtualize get_package_information in chocolateypackageservice virtualize getting package information
replace empty string constant with string empty
apparently strings can t be d
ensure tooltips are closed when its parent detaches from visual tree
use the get or add argument
update nyaa pantsu apipath to the actual feed torznab url
tolowerinvariant for case insensitivity add a few more restrictions
properly theme the password dialog
setup clarify why personal info is needed on setup page
update src features core portable convertiftoswitch abstractconvertiftoswitchcoderefactoringprovider analyzer cs
torznab change content type for t caps to xml only
merge branch master of https github com brunohbrito stackexchange redis into brunohbrito master
look for analyzable views
fix musicstore spa
use compound assignment
add and change xmldocs
fix destination folder in temporary output
includeshared opt in for backwards compat
better initial sizing of hash column which is difficult to resize larger so initial should be good
fix build error caused by trygetazureroleinstanceidnothrow test
fix a possible nullref when login fails
more thread safe method
reacting to hosting changes
updated ssve to support each without a property specified for when the model is a collection
skia throw if asked to render to gl target with invalid dimensions
cleanup dead code
registry prevent crash due to background thread facing problems in the file system
remove an event that is no longer used
delete dead code
change the user name on the caller in chat
add null check for parameter remove unused variable
fix nullreferenceexception when parsing rtf charset_stack was never initialised
ci bump runtime tests timeout after more leak tests
better mvalue formatting in multimodaldistributionanalyzer
test that generated globalsuppressions cs doesn t have multiple trailing blank lines
move to const and add some xmldoc for future visitors
fix update xmldoc to match functionality
add support for combobox virtualization
update staticcontenthelper cs
group all fix buffer tests
trim code down
respond to feedback
remove unused usings
fix loading of settings for build server adapter
make fields readonly where possible
put extension methods in the root namespace
make screenshotmanager a drawable
fix merge issues
add support to force disallowing scriptconfig allowunixpipesyntax false
improve perf somewhat for editor map loading
update version number
include namespace when looking up array types so can distinguish between multiple types with the same name
don t convert originally taiko hitobjects
update obsolete version
because ruleset only exposes hitrenderer we need to have aspectadjust in the base class
fix wpf interop
add xmldoc for onbackbutton
add platformgetmaxnumberofgamepads method to xamarin mac
adjust test to make results accurate
re namespace testscene
more source code comments
password loaded when editing cred repo
use non unicode inno setup in appveyor
update strings cs
fix missed sample
combineddisassemblyexporter output should be aligned to left not right
add specific reason when zero retry attempts were set
fix the social sample port
don t need the invoke here
hounddawgs handle invalid bannerurls
chore remove unused field
squash some warnings
don t serialise path version
updated version stamps in new projects
make username text italic
remove max stack depth
minor style fixes
updated source code comment
fix animations page again
expand comment in createrecoverabletext
fix spelling error in comment
further performance optimization do not render widget view when model is empty
status message ideas
don t use collection initializer in the jobfilterproviders class
fix rulesets not loading in debug builds after running a release build
static class icons in icon bar
fix localization issue with ditheringscale
move spinnernoblink to osuskinconfiguration
show implicit struct conversion
fix the mdr and mt build
fix identityuserappservice delete
fix logo masking not being applied correctly for showcase win screens
use string join in getmessagesrequest
fix grammar in xmldoc
use concat paths in helper functions of googledrive
include start pos when calculating default offset bit size
setup add todo
fix for drawables being empty crashing
remove bogus require s
fix map placed buildings with inotifybuildcomplete
setup focus on entry fields on first start
fix step name and rename test
test update deadlocking test in querybugstest
react to changes in logging missing changes
replace target point by center in all descriptions
fix setting deserializer rootelement from request rootelement
add some more comments explaining why some methods exist
change glowbrush of external dialog after theme change
disable iisexpress tests
remove unnecessary api reference
don t attempt to log multipart formdata request bodies in net core which can t be bufferred
fix decompilation of newarr move array specifiers to the right place
clean up left over test code
fix grantconditiononprerequisite not refreshing when the actor s owner changes
fix bad matching xml tag
better method description
update monogame framework properties assemblyinfo cs
remove unused servicecontroller resolver
fixup serialize_circularreference asserts
ignore test for build machines because wsl not enabled on them
add named params for tuples in another function definition dispose local ivsmddesignerloader in all cases
debug logging only if debug level is true
set nsubjectaltname od self signed certs
fix wrong string unit test
add comment regarding add clear remove overrides
corlib ignore awaited default synchronization context
fix ci issues
increase timeout in server_clientdisconnects_callcanceled to mitigate test flakiness
change order of execution to be chronological
change logevents property to loglevel in latencytimeoutattribute class
update method documentation
remove outdated comment logic
fix xml docs
pass down onjudgement to columns
move humanizing to the model name instead
aviod multiple array lookup
dispose of file properly
remove spurious toarray in mainmenubuttonslogic
remove remaining per update transform in osulogo to reduce allocations
frame bps are temporary by default
add missing asserts
fix the progressdialog s progressbar foreground color when the accent color scheme is used
fix typo in samples
remove unnecessary newline from output make required trait strings a hyperlink
ensure documents are ordered before making checksum
bug unexpected token comment can t deserialize json to xelement back if original xelement had a comment
change github to throw httperror so it s handled like other auth errors
set siblings remove warning
converting args to invariant string
fix placement not working correctly for downwards scroll
examplepresenter display version info at startup
fixup bug do not cast decimal to int as we can miss minimal order sizes
make sure member selector is evaluating target value every time select is made
make some private methods static
react to renames
add validation and documentations for awaitablethrottle maxconcurrent
ignore another failing test
change notmodifiedsince to compare datetime without ms
disable analyzer tests
fix color types in basicmaterialcontent
remove form route form demo project
update gdaxbrokerage endpoint for eurusd and gbpusd exchange rates
use tostring overload supported in net standard
add inverseq to rsa parameters
httplistenerrequestwrapper remoteip now returns the ip only rather than ip port
sort servers with spectators above empty servers
add preservation of the graphics profile just in case the user changes it during the pre graphics device creation event
remove unnecessary schedule and null checks
add test demonstrating issue is fix
fix missing movie search from index null ref
fix potential nullref
update abpviewcomponent cs
test theme add test for default theme background changing
cutoff tab actually working now
fix dumb bug in replay viewer
fix update test
fix incorrect note count probabilities
change sliderwithtextboxinput to transfer on commit
calculation is now a one liner
improve state reason in delayedjobscheduler for enqueued jobs
chooseairfield chooses closest rather than first
getnameofmemberbindingexpression c syntaxfacts
update to getparameterexpression when the source is a unary expression
fix plugin installation
fix invalid cast issue if source isn t visual
fix failure to reset user name and user email during tests
merge branch master into fix decoding loop
merge remote tracking branch gitextensions master into buildserverintegration
remove unneeded async from jtf call
use property instead of reflection now that we ve updated to a more recent sdk
additional logging when we have to callback on language parse
remove unnecessary allocation in c simplify type names analyzer
fix utility mod listing
one mapper instance per unit test
improve the exception message when a bridge s demolishweapon cannot be found
remove unnecessary extra wait frame
remove unused method
use sorteddictionary also in ipc and file read
stop the repair process if the filelists could not be decrypted most likely this is caused by a wrong password
add selectall and unselectall to listbox
android appbarbutton don t apply null bitmapicon this prevents nullrefexception from findresourceid
fix a null reference exception in parentkey
sl core seal managedbytebuffer
restore render timer
remove implicit new combo from spinners
add uripermutationstest percentageencoding_absolutepath
cache delegates to prevent allocations
add osu prefix to mode descriptions
cancel a potentially pending filter update when an update occurs
it is tdestination already
add size overload to resize
minor fix of basictemplateoptionstrategyalgorithm cs
fix the sampler state on sprite batches
fix other regression
remove unnessasary viewport setting
log init errors from setup handlers
add issue links
reacting to disposable logger provider
split nuke announcements based on ally and observer enemy instead of me everyone else
avoid possible steam network fuckups
uwp content loading fix
rename asdefaultservice to asdefaultservices
add initialdelay to cashtrickler
show date of connect and disconnect in raw sample
we can now edit cred repos
implicit string to font family conversion now uses parse to convert
sparklelistenertcp fix reconnect calling socket connect again after socket close causes an objectdisposedexception recreating the socket before connecting fixes this
add tracking info
test module signingcredential update to prevent multiple tempkey rsa access at the same time
fix intermittent test failure introduced by font tests
revert unrelated change
fix editor beat snapping not working correctly when starting with a new beatmap
update baseplugin cs
itemscontrol ensure containers templated parent is null
update ef seeding sample
now has generic action no arguments though
android commandbar fix content rendering bug
add no proxy server example command line args
remove unnecessary using from program homemodule per codejunkie
trigger repaint on visibilitynotify events
avoid indexing dictionary by null key
disable failing test
fix up velocityadjustment value
default getsubcriptions impl returns all security subscriptions
replace new lines with explicit new line sequence r n in unit tests
hopefully fix concurrency exceptions
clean up duplication in timerwidget
update oracleconnection cs fix sql syntax for getschemaprocedures function should have been select instead of sselect
fix removerejected tests
ensure framestabilitycontainer s elapsedtime is zero on initial seek
fix new line
erasure the type of some fields in appdomain so the linker can remove their types
set markdownformat singleton
use combat isinrange instead of float2 withinepsilon for heli
check either bound of the image
update pngencoder cs
fix trait info documentation not being displayed
fix maythrow flag on binarynumericinstruction
emit createresponse when generating ireturn t in typescript
update to listview test not working
use patterns in more places
use pink for sample control points to avoid clash with waveform blue
remove incorrect area sizing now using fillmode relative instead
add bonus mobile build in system data linq not sure if it works well yet
in enumerableex foreach index increment should be checked
admin area accessdeniedview method
event log add todos
fix rare case of requiredindexerflags failing with old newznab indexers
revert explosion condition to strictly less than zero
prevent nullreferenceexception in serviceclientbase constructor
about fix xdg app in version number
improve look of selected match
return an empty list instead of null
speed up findunitsincircle
corlib replace manual array resize with resize call
fix the missing block references prior to looking for missing block info
change colour if difficulty is expertplus
list can not be null in controlpointinfo cs
possible null reference exception
mark immutable field as readonly
use tryaddtransient for defaultobjectpoolprovider
fix doc comment
restore in normal state from minimize to tray
fix nullability errors in jellyfin server implementations
existing episodes will not be add to history
disable non nullable enum parameter test due to difficult to analyze failures
move activity setup to bdl
add missing program shutdown
fix removegrabbed tests
remove redundant call to internalname when mission failed
collect namespaces of assembly and module attributes
add newly required attribute to a test
merge commit caleb master
pass cancellationtoken to getdeclaredsymbol and remove redundant null checks
opengl platform version of mouse getstate gamewindow only works for primary window
fix rebasing issues
fix gameplayclockcontainer operating on beatmap s track after scren exited
make properties private
update with new framework changes and update comment
remove temporary files on startup instead of exit
system net http simplify default http message handler for watchos since there s only one valid value
simplify negative equality expression
changed role of textview from analyzable to interactive
fix occasional class already exists errors
fix ef migration add help add the ability to specify subcommands in the print help command
improve string comparision test performance
fix wrong status after reedeming the key
fix sql query bug
let s test something else
remove using static per review comment
expose span apis on image
try holding the listener open
wasm fix invalid image size constraint
add missing parenthesis
remove trailing semicolon
mono symbolicate fixes directory exists delete race condition
disabled self host tests on mono to stop build failure
add missing interfaces merge error
irenderprocess includeexceptiondetailinfaults true
merge branch master into refactor remove isupportinitialize
add back newline
update resamplerextensions cs
add missing test attribute
fix sampler texture being unbound when using render targets
fix comment also support lang regex
make random in blinds mod the same every replay
add ioc getserviceentry t
it s never null
disable tests failing on clr on linux and mac
clarify method name
removes one culture if
minor rewording of new comment co authored by dean herbert pe ppy sh
make tryexecutetaskinline a bit smarter
refactor configurationbuilder uses ienvironment
populate the iresponse dto with the service response in onafterexecute
use elastic trivia
wasm disable test that doesn t work with wasm
cardigann add support for get login
add more xmldoc
do not walk down subsequent blocks
small comment fix
i guess it doesn t have to be the default configuration
handle invalid msgs not sent with severevents client
gitignore is not a test case
spelling retrieving signed off by josh soref jsoref users noreply github com
fallback to corlib location when sdk path cannot be found
remove duplicate publishdate
fixes default equity leverage for cash accounts
fix unittest different namespace and some comments
use video title set number
graphicsresource fix incorrect comment and spelling error set graphicsdevice member to null in dispose bool to release reference
auto regex add button will use new syntax
disable encryption column certificate error message
fix handling of abandoned mutexes in mutexmultiprocessfileappender
update dlna play to device timeout
fix escaping of strings in po replace with with if not r or n with
remove unused parameters in test
add some tests
scale gravatar icon to look normal at high dpi
fix test dlls being loaded as actual rulesets and failing
call base impl before early return
commented out more stub methods
mono debugger soft handle cases where the instruction list is empty
hiding transient failure
fix issafetochangeassociativity for dynamic variables
validate if attribute is defined before check isany
add ability to exclude specific connection ids from broadcasts add allexcept to hub api
system core update mobile test
add xmldoc coverage of apply
make messagepump constructor public for di
add using pixelstoreparameter for gles
fix individual volume controls not being adjustable via wheel they were blocking each others input unnecessarily
prefer to use the defaultretryattempts const in globaljobsfilter
fix missing w part of matrix in createlookat submitted by sdonze
fix the idle animation of the hostbuilding in rearm cs using the damage state of the reloaded actor
reverts a small bandaid fix that s no longer necessary relating to the order in which osulogo is initialised
fix bug when rewriting expressions with converted enum literals
avoid null pointers when race conditions or faults are present in protocol closing avoid some of the detected faults
properly pass along callback target
only override revert back to original statusdesc if it s not default statusdesc
don t zero cash balances for paper brokerages
fix droid remove multiple partial implementation
scenetime make category optional
add more stock error icons nuke dumb properties i copied from nserror
remove weird time clause
fix broken tests
add copyright header
remove obsolete c tor
when getting season number from paths only look at path file name
fix disclaimer potentially running same code from two different threads
make sure if people await they get a result
clean up code and comments in roslynunsafe
fix the timeline and editor fighting over track seeking
cred manager now recognizes delete key for remove entries
use manual migration
avoid hardcoded hr in stackoverflow test
add comment to workspace ondocumentcontextupdated the pattern matches all our other on methods
moved the capability check out into an extension method to make it clear of the goal of the check
fix searchpanel warning
use count instead of count where possible
skip flaky test
apply color scheme when in high contrast
config save features to config file
detect another new obfuscator
calculate correct sample length when using soundeffect fromstream
fix the folder name case sensitive
size filtercontrol relatively now that we can
fix get ipaddress for darwin
blink install map button
small cleanup in startup cs
disable check while staging the change between open closed
fix layout not being invalidated when re filtering playlist display
fix test case
mention that the event was the type
tighten the example
allow beatmap to populate some metadata defaults if they aren t provided via betamapinfo
use null coalescing operator use null coalescing operator for the repo properties
fix replays not correctly considering negative time diffs
fix license header
fix logo image requests using name rather than id
fix issue with show message dialog not respecting use accent in the demo
make percentagebreakinfoline use formataccuracy
reenable sql azure test that was accidentially disabled in package reorganization
fix build warning
fix build error
system runtime remoting disable test accidentally re enabled by nunitlite switch
always proxy taiko hits when hit
check for org user email to be null
add missing licence header
add invalidateallcaches filter
make all accesses of declaredsymbolnavigableitem s symbol go through the property
make sure the caption bar of a form is always visible
remove dead code
make test public
change cursor according to the command with highest priority
make apiscoreinfo derive legacyscoreinfo
readds remove xagusd currency pair
ignore current type in typeusedbyanalyzer
handle null results in transformmanyblock correctly
s r serialization additional type check for collection serialization is also needed
fix recycling never being performed due to incorrect ordering
use dognzb name as the default rather than the url
make default stackpanel orientation vertical
allow model level ruls with validateasync
update servicehubremotehostclient cs
corlib fix test to not rely on compiler generated method
combine conditionals and clarify comment
comments add remove for menucommands
local functions related ci fixes
fix the doc comment for messageresult
override getservicegateway in basicapphost to use basicrequest if null
remove stray console writes from tests
conditional defines should only be add once per symbol
remove unnecessary type specification
merge remote tracking branch jbialobr copiedfilesdiff
fix an issue where rootpaths with a slash at the end would have double slash when add a new series
fix for directx cs
clean up duplicate trait lookup in raspecialpowers
bleep when objective shown
further changes on system info page
add another failing popuproot test
fix crash due to misordered selection events
fix crop issue
chore ignore ui test
optimize caching the number of comments to news
isvisible patched to know about gps
fix real ra map importer when map isn t in game root
missing webutility methods
make editor discard approach circles temporary solution for now
add support for old beatmaps
add more documentation to the autosize property
fix a incompatible readonly textfield color with the netdriver if its selected this work with all drivers
remove extra return statments and add to new paths during rename
fix potential off by one
fix failing test
display important error logging in notifications for now
move multiplayer leaderboard test to correct namespace
format exceptions in host
disable subdomain normalization
hide the avatar control when the showauthorgravatar setting is off
ensure a valid target in move
remove canentercell from occupiedcells
remove temporary code
avoid printing the output of bin ls in a process test
remove completed todo
updating derived collections are now not scheduled
use hostname environment variable for unix
prevent catcher from performing invalid catches
update gdaxbrokeragemodel failing test
prevent log file writing when option is not set
merge remote tracking branch origin develop into mit license
remove unused code
add setvaluetranspose to effectparameter class
use setupeditor instead of settext this change updates the initialization of basicnavigationbar to match the behavior of csharpnavigationbar
use readonly ibindable
use shared implementation of transformtovisual for unitests
fix wrong parameter
fix compilation errors in tests
remove default value for memory cap to force explictly setting
refine debug versus release use of timedlock refine debug versus release use of timedlock add detailed explanation for future consumption
avoid semantic model lookup and filtering for explicit interface implementation
sound effect volume apply master volume at correct position
renamed inheritance copy to clone
remove unnecessary using
fix typo in comment
add browsable false to editnetspell s currentline and currentcolumn properties
fix scoreinfo not getting set
add subscription exception logging
adds a missing tolist in assert code
asynclogeventinfo remove private setter
add simpleteleport for non chrono jumps used in scripts etc
disallow negative zero repeat counts and fix off by one
make uog actually use findunitsatmouse rather than homebrewing it
try again on bad merge
linux setup fix inconsistent phrasing
avoid an unneccessary abbreviation
if http charset header is empty do not append content encoding to content type header
show on test case
fix attempts to start broken bots
fix response cookies namespace
add file path
set useshellexecute to false in get help
temporal vortex attacked again
enable working test that was blocked just by xmldocument api
address cr feedback
move to new test harness
fix command will tell you to press any key
work with repos on host names with a dash in them
add botdebug message for external unit build requests for easier bot debugging of things like mcv and harvester replacement
improve performance on distance calculation
vs designer needs this class not to be abstract
fix failing tests
pass a cancellationtoken when we have one
oops forgot the constructor
increase retry count
unskip a test
add clear bool for free dataslot
fix inertial scrolling
adjust tournament map pool layout to allow for larger pools
update languageservicetests cs
cancel previous task
fix bug causing to not show gitextensions on taskbar
changed notification from installed to imported on beatmap download
fields can t be immutable
add named parameter
fix items with iconbrowsefileexplorer
add tosafedictionary ext method on ienumerable cause the default linq is stupidly fragile and only on list
merge branch feature convertnameof of https github com m redding roslyn into feature convertnameof
set a busy timeout when creating an sqliteconnection
remove unnecessary using declarative
make tests not fail just because a same named env var was already present
start line with boolean
remove android touchpanel hack
make synthesizeddelegatesymbolcomparer instance readonly
toolbaroptions to abptoolbaroptions
readd event handler remove from merge
corrected where statement because if its provided it doesn t need a parameterless constructor
chore adjust sample using
remove irrelevant check for parameterized querystring
add copyright header
system web fixes skinid property declaration
fix the idle animation of the host in withrepairanimation cs using the damage state of the repaired actor
integrate timeline into compose
re enable async lock on distributed cache
task from result in examples
adjust slider fadeout slightly
use correct tab name for whispers
add quotes into error message so it s a bit easier to understand
add todo to remove string target
designer redirect standard error
add default argument
forgot to commit tests
move incorrect placed full stop co authored by bartłomiej dach dach bartlomiej gmail com
ensure selected dropdownitem is visible when dropdown is open
update logentries token again
trigger graceful shutdown on stopping webhost service
use the minkey for new cursors
only wanted is default for cp
add workaround for post limit size
remove position offset from columnat
securestorage remove unused exception param
client socket close add
fix downloadfile when file already exists
avoid unnecessary object updates from selectionhandlers
remove cancellation of linked tokens
fix panel selection flicker
remove unneeded lifetime assignment
fix crash when starting with an empty bracket
minor cahcing issue fix product breadcrumb wasn t changed after chancgng a product name until cache was reseted
use ascii mode for measurement presentation in terminal
fix mania frame conversion not working at all
add number of cards drop remaining in status response
fix a tooltip crash
discard a reference to project being emitted when debugging session terminates
fix id s of missing episodes
remove leading i char from interface based jobs
fix string concatenation
complete the rename of two wekkknownmembers in comments
temporarily disable revert functionality for audio device since it crashes
show warning on editor when editing git config
fix crash in editor when trying to seek on a virtual track
merge remote branch vasily kirichenko unique get files
fix tab close
fix double release transaction in query error
update client id secret and bring api endpoints up to date
fix namespace in code references in xml documentation
merge branch release into dev
use occupiesspace to save more trait look ups this time in locomotor isblockedby
correct assert message make private constructor
fix error where jobs werent ack d with a runtime error
fix tiny tiny nit
remove unused using directives
prefix identifiers starting with invalid characters with an underscore
randomize people task default start time
react to logging api changes
fix random layout animations when loaded
use more indentation on wrapped lines
fix doc comment in selectexpression
install typescript with npm
header for backgroundjobserveroptions
update formsauthentication cs
use tuple deconstruction to swap values
fix up texture2d contrustion for winrt
tidy in selectable
only switch the system proxy off if we have switched it on
redis integration tests fix
add nfw to help track down issue
fix bad env var from being created when pythonpath is empty
don t cache useless data
browse delay refresh stash list
add test illustrating issue
do not convert property if the getter has any attributes
make the authhttpgateway urls static so they re overridable
split out queues for live data feed
update csharphelpcontextservice cs
update program cs solving the problems that project cannot be run on iis
actually implement the interface
fix fully qualify window reference for ios
fix unit test failure
incremented abp testbase version
we don t need that delay anymore
improve unit test
stripping dead stuff
minor code cleanup
reacting to korebuild changes
update myprojectnameamenucontributor cs
remove unused using statement
delete unused compileranalyzerconfigoptionsprovider isempty
move keybind to last in enum
add support for country rank
split noto out into individual ranges
fix invalid value
merge branch master into channel topic serach
fix silverlight client
iswinplatform was written twice
fix compilation for silverlight
fix missing using in functional tests
fix localization problem
use minwidth on pp column
fix symbol test
update unit test expectations
self document crushclass of crates
fix macos tentative performance update for nstextstorage
whitespace change to retrigger a build
fix merge conflict
add missing ctor parameters back
remove unnecessary connection factory instantiation
improve accessing responsestatus prop perf
rename microsoft aspnet http interfaces microsoft aspnet http
getstringdecryptermethods now adds all string decrypter methods
enabled exception propagation tests
produce correct error
add explanatory comment
fix right click regression
add load dependencies command to assembly node context menu
prevent nre when updating hidden panel
remove redundant get set
allow uwp icon extension to be changed
filelist add support for banner url
add missing compiler directive in unit test
react to breaking change in mvc cc javiercn
switched the logic around for an early exit to avoid a local variable
print full guid in guidid
don t run render tests in parallel
improve error handling for registrylayoutrenderer
bump assembly version
fix infinite loop in testdriver cs when using the v option
fix fanart reference
fix brush wasm update frameworkelement background when solidcolorbrush color changes
make setdataannotationattributeerrors only called once
fix compiler warning by including raisepropertychanged code in default tile implementation
use await instead of wait in test code
attach styles in reverse order
fixes mistypos and marked useanimations as obsolete
add a check for non usd currencies while updating cashbook
set encoding for omdbapi calls
enabled beta for testing
fix fileshellsettingsconfigurationprovider lifetime
tweak async modifier constant value
making bytesize iformattable
updater workaround for waitforexit
fix the coreclr contract name for portableshim xpath extensions
remove redundant transform initialization transform is already set to matrix identity in the nodecontent constructor and meshcontent inherits from nodecontent
fix potential deadlock during gameplay tests
change add to to collections doesn t make send to be add to when it can also remove
exclude the p invoke code that is not supposed to be in coreclr build
remove useless calls
chore dragdrop update comment in test page
fix equals mistake
add auth headers
went a bit too far
fix charset setting for jsoncontent
fix avaloniaproperty registration type
fix an issue regarding silverlight and windows phone clients weren t sending cookies on request
remove old test
add item flyouts
update src visualstudio liveshare impl abstractgotodefinitionwithfarhandler cs co authored by cyrusnajmabadi cyrus najmabadi gmail com
use string concat instead of
fix console logging typo in mirrorprovider
add tooltip text to results screen pages
merge branch master of github com servicestack servicestack
update strings cs correction spelling mistake
disable com specific stringbuilder marshalling
add better support for downloaded dictionaries
write a message if decryption probably failed
skip online test if user is not logged in
merge branch master into eisber master
don t print stack trace to console for invalid config files
test cases ignore text
cleaned up update method in taiko playfield
minor clean up
add a convenience copy constructor
update featurevalueconsts cs
prevent binding on some columns in order
updated automapperstartuptask to ignore shippingsettings usecuberootmethod property
bump copyright year
minor formatting no real changes
merge branch master of https github com massix sparkleshare
make method public
move usings into namespace
fix hopefully last build error
simplify pushposttransform usage
merge referencecounteddisposable constructor behavior
fixes failing test
clean up render timers to use environment tickcount
handle null in getdirectorypath
use async call
update dnr detection
add a default icon when a ruleset isn t present
don t remove any types methods etc if it s an unknown obfuscator
use the correct variable
fix stage background being too transparent this is the original colour for the background not sure why this was changed
chore address suggestions
make sure ai can deploy an mcv when it lacks a base
remove unused inotify interface signed off by chris forbes chrisf ijw co nz
add comment to selectmany
useshellexecute false to redirect setup console output to current output
remove more vb referneces
add method to remove all entries for a folder
make childproviders public
align swellsymbolpiece better for rotation
add missing xml comment and moved using statement
test fix we don t let exceptions go unhandled even for built in analyzers
auto adjust menu to contents and set minimum
add inheritredirectclipboard to csvserializer
fix portfoliotarget percent test name
optimize selectall to make it as fast as ctrl a
update usermenuitem cs remove privates set other ctor from internal to public
change to use custom exception class
fix infinite loop in astcomparer
fix routing sample
update src features core portable conflictmarkerresolution abstractconflictmarkercodefixprovider cs co authored by sam harwell sam tunnelvisionlabs com
avoid including stray commas in hls codecs field
check core dlls when running leproc
fix extension method
fix broken swagger tests
change to until step fix ci error
fix playback position being set incorrectly for ianimationtimereference
remove unused decimal icalls
register also default assemblies in loaded files list
reduce line length
moved the hubpipelineextensions into the core namespace
remove unwanted waithandle finalizer
add warning message for invalid remote control port
add enabledbydefault check to withinfantrybody
reacting to logging changes
fix ios build
make pngbitdepth a byte
consider only webproxypassword to be sensitive
fix argumentexception that was being thrown
update encodervalidator cs
abpcomponentbase should be abstract
web fix obvious typo in delete from where statement
merge branch master into no drawable speedchangevis
fix description for repositories on dashboard
fix license header
rename class to streamxxx if it s a stream class
use proper os comparison
add wangle lerp function
merge branch master into spin in
fix duplicate collectionbehavior attribute
max depth working
change infinite loop to not to throw exceptions
add urlencode urldecode
throw exception if specialcolumnposition is set after isloaded
add missing xml comments
update embedded metadata extraction
make apichangelogentry major a non nullable property
fix guard agains null reference exception with newznab capabilities
add insertafter insertbefore to astnodecollection
clamp selected preset index
fix an unrelated editor regression
add project scope fixall test
fix binding mode checks bindingmode default should be equivalent to bindingmode oneway
merge branch dev of https github com aspnet entityframeworkcore into dev
fix metro cert verification error
nlogtracelistener disableflush should be true by default
rename test methods
trycreate uri first
nulleventbus is made singleton
restore old test add new test
cr rename rendertreebuilderextensions to webrendertreebuilderextensions
fix minimum width for the panel and the form
remove public add static
hack belowunits z
revert to cloning to sparkle tmp path
shortcircuit processstartinfo verbs logic let s not do io operations except if needed
add unknown fragment test case
add x generator to part of the shell middleware pipeline
fix instance where sonarr was used in logging message
use safe accessor when creating boundconversion
manual property on command resource
fix build of testplugin contextmenucommand
fix decimal display of beatmap details being too precise
add adjustable clock to testcase
remove unused function
no need to process result if timed out
add cursor rotation on middle mouse button
fix dynamic serializing in silverlight
always set file menu position to the top left
cardigann add support for redirects on download selector
react to ef change
make handler shim abstract
dont make native handle null so closing events can complete
fix client scope access
merge branch master into fix linux main loop
making public apis virtual
windows setup swap finish page buttons
set the default voxel scale to match the original game
fix ismaskedaway not being up to date when fastfowarding replays
merge branch marcelroozekrans master
move iapplicationdiscriminator to infrastructure namespace
extend showdeclaringtype option to nested types
fix and add timespanwithprecision tests
update metadatatypes cs
changes null behaviour and fixes test
validate chapters coming from xml
further changes on the previous commits minor performance optimizatio
fix hotkey dispose signed off by syrone wong wong syrone gmail com
add argument name to argumentoutofrangeexception
updated that the base method is a noop
xbuild evaluate import only if its condition evaluates to true
dont crash on invalid dll tested it local
moved using statement inside of namespace
fix the build
update logincommand cs
moved the iauthenticator into the security authentication namespace
fix silverlight build errors
add flag to enable disable launching of ib each test
don t log discord connection failures
setting up constrained notification handler example for simple injector
only enumerate crushable traits once in mobileinfo isblockedby
add static flag to be able to disable flushnoponsubscription
more parsing tests to ensure anime doesn t break standard episodes
add framework target var to buildboss
small viewengine bug ensuring not null
chore styles comment out noisy assert in navigationview
merge branch master of git github com chrisforbes openra
fix pathing to repair pad being bogus
fix ci issues
remove incorrect assumption from tests
merge branch master into improve volume controls
system fixes failing test
webresponse cs implement supportsheaders property
improve handling of concurrent analyzer execution
remove duplicated check
add beatmap argument
rename test method
change servicecontroller execute to execute tasks synchronously
unify join time text s visual format with the web
query won t need translate since query will always be engligh pinyin
fix saveremote tests now that module addremote is expected to return empty string on success
fix openapi tests
make sure a treenode is selected after changing assembly list to make commands work properly
add overload for urnid create to take an object we tostring on
symmetry between reading and writing int2
fix d rank displaying as f
adds multiperiodfield hasvalues null check in updatevalues
ensure correct initial state
bind to correct target hacky
mono api html fix ending quote for non string fields
accept devcommands only from local players
improved verbose messages when excluding a file based on the size criteria
simplify implementation play fail animation during restart
improve rename vb test
catching all exceptions
fix override screenshots folder issue
etag generated in embeddedfileresponse surrounds the value with quotes
make seal last and move newing up create map functions before do config and seal
don t connect the source observable twice if we re in a unit test runner
fix xbmc cleanlibrary
chore inlining basicgeoposition from peer review
readded public to aesstream and tempstream
move copying down
cancel old web requests when changing filter modes
re enable valid tests
add a dbgobject dtor in debug builds
hide windows specifics behind an if
fix code model implementation
handle onhover on song select filter and footer
add requireexplicitimplementation to the icrushable interface
fix the build break
add a new random name regex
controller fix opening links with spaces in them
fix spelling error in crate comments
only show random skin button with more than one skin
fix lookup cache throwing a null reference if no matches were successful
integration tests for swagger ui
update github url used by get api token
fix equals for null value comparison
make it clear that maxclientsize is used when available
disable a test which fails with aot
fix comment in sourcemembermethodsymbol verifyobsoleteattributeappliedtomethod
update csharphelpcontextservice cs
fix crash when selecting non existent ids from database
avoid output when disabled
disabled resharper unused method check
missed a word
revert samples program cs
pluralise playlists in tip
filelist update passkey info stress the difference between password and passkey
fix merge typo
fix fxcop build break
mark grid layout methods pure
prevent restoring window styles from overwriting windowstates
maint formatting wording
shorten main namespace
update repos after succesful cloning
add messaging to streaming and and ux for when config not set properly
merge branch master of https github com mediabrowser mediabrowser
add note about substring usage
change net core naming convention default to emitcamelcasenames true
remove unnecessary code
fix asf ui for single file publishes
don t return plugin versions that target newer jellyfin version
fix collection grouping for plain folders
fix warnings asking to use await
add tests from chat with grokys
remove extra return statement
made abpsignalrcontractresolver ignoredassemblies static
ignore ss client mega dto test
update cliservice cs
make registrationfeature atrestpath user configurable
generalize data type used in slice getkeyvaluepairenumerable
initialize fatal error handlers in vs and oop
fix resharper warnings
add missing protected ctors
this doesn t need to be so specific
make runbase wait on previous sends
fix open output in ilspy context menu item to focus the assembly in ilspy tree correctly
write web config before setting up iis to avoid app domain restarts
reduce horizontal spacing of attributes
add support for converterparameter to multibinding
clear previous score items on setup
fix alliedbaseproc unused variable warning
updated a property comment
disable two delegate tests on mobile
some noise on shellmap
add disablekeyrepeat flag to buttonwidget
fix test after rebasing
fix update test fails due to throw elimination
fix for sqlserverconnection getstatedata query
scroll to screen middle
fix unit test
fix skin changed events triggering after disposal
move sliderfollowcircle implementation
remove unnecessary attribute
adjust header padding
add getobjectid and createurn methods on idutils to get ids and urns on late bound objects
windows sometimes strips protocol when url is passed to handler conflicts sparkleshare windows sparkleshareinviteopener sparkleshare invite opener cs
add forcemanaged to createfileparameters
don t call completereader the dispatcher can handle that
fix parameter reference in anonymous function would occasionally generate exception invalid attempt to call metadata when reader is closed
add a do nothing interface for opacity masks for skia
make control template of type icontroltemplate
fix sync job deletion
ignore esc text input
remove deprecated maxloginattempts
expose the itracemanager as a first class property on globalhost
update collections folder
use milliseconds for basicstats beatmap length
update connectionmultiplexer cs
better way to set default directx apis
remove incorrectly add key
disable warnings for tests of deprecated apis
same for smudgelayer
chore shapes fix merge conflict
compare item in revision grid context menu now uses diff icon
fix comment typo
fix storage fix invalid refactor co authored by jérôme laban jerome laban nventive com
delete more environment variables co authored by sam harwell sam harwell microsoft com
we pass the destination member
system test processmodule managed assemblies
move variable outside of if block
fix factor files missing file dir check
mark tokendata as a readonly struct
add xmldoc to musiccontroller resettrackadjustments
entry removes some lines there were commented out
progress bars on resumable items
system io compression do not swallow inner exceptions in ziparchive
fix wrong beatmap selection in song select menu on deleting selected beatmap
handle null bid ask in quote bar
add getpermission overload
prevent hang in itemscontrol with no focusable controls
fix change namespace service to deal with referencing in razor document w o crashing
fix rangemultiplier regression
use compund assignment
add more gateway enums
cleanup isgenerated code check in csharpsyntaxtree
fix switch case caused by a poor merge
updata _datatoken to _datatokens in virtualpathdata
cleanup event args locals
minor comment change
don t check hasadequaterefinerycount if no refinerytypes defined
change hostcontext resolveservice to accept httpcontextbase
remove redundant base call
fix update movie libraries in plex not series libraries
update cacheentry directly
attempting to fix ci
fix a typo in comments
make idle timeout for idle threads configurable
merge branch feature ffmpeg version check of https github com rotvel jellyfin into feature ffmpeg version check
add devtoolsurl to iwebbrowser
display visual settings overlay on playerloader screen
fix a null reference exception
update diffcalc test
increase steam launcher wait time
expose overridemetadata for direct properties
add test for asqueryable
fix bug in skia empty formatted text hit test
calling dispose multiple times should be valid
fix valuechanged events being called out of order
rename local function
fix merge conflict
add company copyright and product attributes to assemblyinfo
remove unused runbackup method
preserve save session during oauth flow
fix search field losing focus
workaround mono specific problem with uri trycreate
paypal standard namespace issue fix
update commentingviewcomponent cs
check for commits only
rename intrinioeconomicdata cs to intrinioeconomicdata cs
fix some nullrefs
don t eat exception from timer handler this is system timer not swf timer
re query beatmap difficulty before computing
handle cases where a namespace declaration isn t found in the ancestors of a type declaration
revert failing parsing tests
windowsidentity instead of principal
handle null strings
merge remote tracking branch origin release into dev
fix compile error
don t handle content requests in indexhtmlmapper
clean up some garbage
remove reference to usedefaulthostingconfiguration
add unit tests
fix bug in wrappanel invalid size returned for measure
fix typeofresolveresult in attribute arguments
remove unused usings
dead things in sheet
fix test file casing
try out another way to block
auto organize updates
fix hitobjects in scrolling rulesets getting masked away
contentproperty will just show up as an ordinal property so no need to worry here
this compression is actually not yet supported
distribution script bug fix
fix history recent events jump to top refresh by blocking webkit from loading uris directly
fix an issue where an unloaded movie could case linking to fail
fix mania hold notes displaying judgements
thepiratebay drop author author is used for music media results and is not the name of the torrent uploader
add copyright header to some new files
fix htmlformat build error
make exit s descs more informative
remove frame requesturi
fetcher make snet repos use client side crypto by default
make osutestcases use new non realtime option
implement remainder of matrix interpolation fallback test
chore adjust waitforload check
update namingstyle cs changed resource string for error for checkalllower
remove braces co authored by salman ahmed email iisalman me
oops that crashed when there were no mods
add test validating scenario is fix
handle formatting for datetime inputs for extension properties
remove unnecessary line in jobcreationprocess
remove unnecessary check
intro add cancel button to add dialog when needed
increase the number of repeated run in stress test
remove used of assemblyresolvestategy in the razor engine
clean up code a little
change request method
re enable benchmark
load seasonal backgrounds according to setting
allow osx to run in vm where opengl init may fail
mac ui update to san francisco font
fix off by one starting bar
fix plugin security saving
merge branch master into refactor selectionmodel rewrite
implemented this new executescript method
fix spelling of breadth
use more correct equality code
fix bad format string
exposing the installid for the comming usage reporter module
add thread asserts
use local variable
update code since getuserstring arg is now a token
merge branch master of https github com dotnetcore cap
minor refactoring iwebhelper issearchengine method now accepts httpcontextbase instead of httprequestbase
on freehand region step drawing resize call move method so arrow keys can move without require holding ctrl key
follow code review suggestions
update some colour palette discrepancies in options
always refresh markers even if they re not visible
remove stack trace from serializationexception message
fix expression evaluator unit tests
ensure that exe projects specify an applicationmanifest property
windows setup add history checkbox to add page
merge branch master into reduce array allocations
remove duplicate query
fix skinfallback test crashing on repeats
fix unnecessary assignments
remove unused field and documented refreshchannelsscheduledtask cs
remove dead code
fix nre when handlers defined in location tag
assert single channel support for now
list all spa project template names
add support for instance explicit calling convention for a methodreference
arestringvaluesequal for real
edited sparklelib sparklelistenerbase cs via github
fix leak by releasing the certificate elements from the cfarray since it was created with a null cfarraycallbacks
correct exception type
add padding to show more button
use extension method syntax only if the extension method is eligible
add reason property to entitychangeset
merge branch featuru pattern foreach via extensions of https github com yairhalberstadt roslyn into featuru pattern foreach via extensions
make ireadonlyperspexlist covariant
add dotnet_use_polling_file_watcher to help output
replace default clone with createdefaultconfiguration
add missing class
fix logo heartbeat playing even without beating
exampleresourcerequesthandler add a brief xml doc to the class
fix mouse coordinate for desktopgl and windowed mode
use linked cancellation token
switch device type around to match mobile enums
mgcb editor windows sizing fixes
fix async warning loaddebuginfo
fix project build error
convert switch to if
add parenthesis to fix osx
refactor restrict code into more logical structure
require explicit igameover
fix nullref in tests
putting repository stuff into subfolder
allow recovery from a very broken database
avoid accessing beatmaps from beatsyncedcontainer until they are loaded
make sure icon is only instantiated once
fix namespace and unnecessary using
additional check on module discovery
main fix compiler error on linux
avoid redundant property set the property already has this value and the setter constructs a bunch of other objects
make isvaluetype actually return isvaluetype not the opposite
explicitly import immutablearray extensions
check for null when creating subscriptiondata
fix nrt error
ability to pass multiple assemblies
add file header
remove response code doc because it got add elsewhere
remove unused property
use datetime utcnow not datetime now in test
combobox fix crash
remove unnecessary null check in dataqueuefutureschainuniversedatacollectionenumerator movenext
remove old crates list
remove unused property
moved config window localization into the ctor
singleton instance should be given focus
update disasm tooltip to include cpuid info
mark test as ignore this test fails sporadically but passes everytime its run by itself ignore for now
add a helper
mark regressiontests as travisexclude
conventionalcontrolleroptions to abpconventionalcontrolleroptions
fix bug of unityasyncextensions ugui can not handle remove event
incremented version for the release
seems to work properly
adjust mania hp increase values
fix platformgetplayposition for wms
further changes on the recent commit related to baseprice pangv
fix isourcedfromtouch events being blocked by loadinglayer
windx small fix
fix ai jam in cnc
ignore distance for json serialisation
disabling query gating
remove another tostring statement
improve the documentation comments on codefixcontext constructors and codefixprovider registercodefixesasync methods
make string comparison use ordinal binary sort rules this string comparison should not be culture aware
chore scrollviewer fix ci build
test only on ios and android
place rim hits using right mosue for now
start using stoptimeoutms
method no longer used
add contractannotation for null check
update trace message
make more threadsafe
updated mdoc to remove assemblyversions tag
merge remote tracking branch upstream master into combobreak
new nzbgeek add as preset indexer
remove unnecessary checl
fix add flac to mediainfo renamer audio codecs
fix build server settings page that could be clipped
ignore errors getting telemetryhelper
support single tmdb urls in nfo
fix an issue where scheduled times would become offset if the user is in a non utc zone
fix fastrandom using uint instead of int for nextdouble
fix null issue
update localizationservicefixture cs
cleaned up using statements
add record support in lsp
work around httpclient tls regression
update custom uploader link
fix potential song select nullref
update enum name
merge branch master into fix direct tab activation
smtp env config
remove more garbage
fix hide inherited property
remove need to trim query string
remove unnecessary mobile local
move imultitenant to volo abp multitenancy namespace
revert fix decompiling empty async methods
add realistic data for unit tests
use expiry to avoid toarray
use component instead of drawable
reordered parameters in new method comment out stuff
revert unnecessary change
setup linux fix grammar
updated xml comments on ifluentadapter implementations with wrong description
stop using obsolete variable
resolve a bare to the game install path
fix class reference in xml documentation
remove unnecessary casts
improve for the backend web preview
improved cleanup element handling so it is not set to window if set to something else before
spaces to tabs
remove out of date comment
remove unused fields from designers
fix contention during scrolling
merge branch master into confuser
update diagnosticresultserializer for analyzertelemetryinfo concurrent
make sourcehook virtual for wpf chromiumwebbrowser so inherited controls can override custom hook
update hotkeyselectioncontrol designer cs
http disable redirect with body and no buffering
don t pass in any options
fix actor currentactivity bogusly pointing to finished activities
make code prettier
update crudappservicebase cs
eventlog remove debug line
need the explicit inheritance for correct intellisense
account for initial health
fix osu direct hotkey overriding drawings hotkey for now
changed metro build to use late bound reflection
render ore when gps is enabled
fixes spelling in comment
use interpolation function directly for resizing tab strips
made the link s onclick method return true
add more checks and remove direct cast
remove unecessary convert
make ireloadmodifier require explicit implementation
empty collection check
add back default content fade transitions
make resource warheads airthreshold aware
restore tests that were testing obsolete members
react to deployer changes
avoid closure allocation in completestrackingoperation
fix failing test
add test ensuring correct spm calculation on rewinding
add back parent child linking
simplified save filter a bit
use faster string checks
forgot last whitespace
stop the boom
fix getstableimportpaths xmldoc
use fade instead of negative margin
adjust visual behavior
fix service registration order
remove opt suffix since we have nrt
fix regression in suppresswarning
replaced mobile with explicit android ios mark
fix special power bin
add windows phone exclusion on cancellation token async stuff
disable hd for taiko
update xml doc
when region capture form focus lost disable modifier actions
fix glitchy vis on exiting a transport
fix populate originaltitle in db for listmovies
fix stylecop warnings
fix shroud rendering outside the map cordon
feat add temporarycredentialscachekey
fix up existing messaging
fix problem that building string took an extreme long time when a command has large output
fixes bug in matchinglow pattern
fix default turreted rot
use equals instead of ref compare
add support for client certificates and the server certificate validation callback
don t serialize totalbreaktime
test null graphicsdevice spritebatch
fix scaling of greyscale visual obsrrvations
additional code cleanup
don t crash on unimplemented sorts
don t need explicit null value
exit import process after first handler is run
pipeline tool fix context menu for non existant items
fix skiawpf mouse wheel was too fast
infdeath deathtype upgrade rule fix
fix the namespace of some corlib tests
remove compiler directives
remove unused usings
move into components namespace
changes to implement the new resume method used in soundeffectinstance under android
no need to owner group change check
add maxlimit to autoqueryinfo
pass damagetypes in explodes
remove unnecessary usings
disable warning about unused class
comments in test dynamicdictionaryfixture cs should_remove_natural_key
add disposition header for csv script responses
changed dblocator so it does not use the default backup name as the db name
fix some caching issues
remove rogue using
mcs constant folding of null strings casted to object
rename function for consistency
fix incorrect fallback index being used
fix missing comments
temporarily disabling failing iisexpress smoke tests
fix rounded corners one pixel off issue
update video params
fix typos in comments in roslyn workspaces workspaces casecorrection
simplify path construction
used wrong type in equals
update file names to better formatting
feat translators add little bit more intelegence into varchar detection
update entitypropertychange cs
make ground actors leave resupplier if resupply is aborted
gitui hotkey keysextensions toformattedstring is now able to produce culture specific strings
changing test filename add prefix to filename to guard against interference between tests
fix another hashed release format
fix format for standardhtmlusermanual
update components to use extension methods
disposed of cloned stream after usage
fix osu supporter naming
add failing test itemspresenter items isn t correctly in sync with itemscontrol items when assigning new items
remove unnecessary assert
modofiy class access level for diagnostics
use configureawait false when awaiting unionwithasync
fix up the tests
properly unregister broadcastreceiver on android
fix tests after remove sentry logging
replace bindto with setting the bindable
replace invalid characters
repo align regexes
autosize notifications and add animation
reset the listview when the source is defined
update rename regex
missed a psm draw call missed a psm draw call
use pattern switch
do not hardcode status codes
git fetcher fix resolving conflicts in encrypted files
fix sql error on downloadhistory insert
remove the cleanup phase
itagproducers don t need to be disposable no producers ever take advantage of this
move createlayoutmanager to be next to other methods
mac watcher also check the timestamps of files
remove test patch
possible null reference fix
keep some stuff
remove unnecessary string interpolation
baseindexer update cookies in requestbyteswithcookies
use unified name instead of full name handles class t better
fix memory leak
correct hascustompluginsloaded check
norbits fix download links
onsettingschanged is now protected
rename simplebundlecontributor to bundlefilecontributor
clear items when items changes this allows us to work with lazily evaluated ienumerables
fix expiry date cache query in ormlite cacheclient
razor views are associated with cshtml instead of razor
return download link if the file is not an image
merge remote tracking branch origin master into fixes nre menu interaction
if client rectangle same size as window size then don t add it
change derived class to use property
fix info wedge not displaying
renaming commandkeyslot to keyhashslot
fix the parallax background missing from the space occupied by the top toolbar
fix wrong path of bgr api
updating issue numbers for some skipped tests
reword and add missing xmldoc to onscreendisplay
last fix to ensure loading correct settings file
fix a race in task waitany
add missing await in grid integration tests
add more assembly search paths
add test for string concatenation operator with mutable value types
use tracking to find nodes after the change
make beatmapinfowedge present until info is not null
fix charset in json tests
increase spacing to closer match the design
add missing generic mappings to legacy mods
don t fall through to seek if handling zoom
assemblyhelper protect getcustomattribute in try catch
null check for translateform
logeventinfo hasproperties should allocate when hasmessagetemplateparameters is true
fix stateless jwtauthprovider tests
fix teamcity build chooser the id of the project should be used in the rest url to fetch builds
use explicit typing
use env contentpath to resolve files in net core as works w publish outputs
stringtable does not have a default constructor
make colourprovider accessable from derived classes
add clearer exception message
allow the chatlinktestcase to work again
rename incorrectly named container
fix build signed off by dotnet bot dotnet bot microsoft com
osx avalonia native fix nre in double dispose of screenimpl
add jsonbody extension method so that you can send a json body request
fix withcoloredoverlay upgrade rule
add old string concat with __arglist
fix incorrect cropping in fonttextureprocessor
adds produces annotation to the base controller to indicate application json as the response type for endpoints
fix catcher s catchable width being half of what it should
add the timestamp property back to the version info
merge branch develop into fixspritefontsize
remove new line for codefactor
create some global constants for cache limits
fix song playposition that is invalid on android after a play pause cycle
change options to post and call saveconfiguration
revert negative only time difference values
fix formatting nit in sequenceprovider
updated source code comment
linker allow accessing more markstep mark methods from subclasses
set url on load start
fix system nullreferenceexception when playing dvds copied to hdd
fix cinema mode failure
update parameter tooltip icon
fix powermultiplier namespace
simplified typetocompilablestringextension generatearraystring
update focus requesting code in searchtextbox
fix missing rules in prep update path
improved speed for creating the scantime lookup table
remove second unnecessary colour set
disable two tests which fail on windows with the net runtime
fix max view distance check
force tooltip to close if the datacontext and therefore the tip property change
fix xml documentation for partial class logger
remove liveshare comment in trygetcontaineddocument
test adjust unit tests retries only for runtimetests
fix extension substring issue with static content
picking the right event for propagation to user algo
add local endpoint
correct doc comment
country palette etc works now
add xml comments
fix post merge issues
ci fix build
bring up to date with latest framework
make chevron icon colour protected
remove code commented
xml fix getattribute to handle null namespaces properly add unit test
fix traveldistance not being calculated
adjust mobile so that it also can t double add to uim
remove spurious allowmovement parameter from scanandattack
let mobile s canentercell consider tosubcell
update assemblyinfo cs
add findtargetbyname with type
add a predefinedcommandhandlername
clean up header qvalue parsing
fix infantry idle animations playing immediately after creation
update csharphelpcontextservice cs
add abpcontroller validatemodel
update authz policy to new scope name
remove unused namespace
fix a nullreferenceexception when merging dynamicparameters
singleton at end
trim header comments
prevent revert to default onhover from hiding visual settings at beatmap load
fix razorremotehostclient to maintain binary back compat
use hasreturnvalue method
chore scrollviewer code cleanup
rename resource string
exclude for sl clients as well
announce map changes
move hardcoded playeractortypes to shared const variables
cursors for engineer
move score nulling out of loading property
implement candeserializeintoexistingobject for listwriter
use datetimeoffset with fix time zone in tests to allow it to work in non pst timezones
fix error handling in do while
delete some dead code
initialize result handler with algo sooner without this errors thrown in initialize aren t sent through result handler
add missing variable declaration
add method to check if iglobalmoddata exists
add extra tabcontrol test
actormap is updated in tick so we can t search for adjacent actors in tickrender
use explicit null instead of default
allow to extend the claimtype
use stringbuilder append instead of string concat
fix typo in content localization
make usesharedconnection available in tesstconfig json defaults to true
add quotes to netsh command parameters
reenable functional tests
allowed media types for icon images
should check ismappedtofieldforefcore
don t force public if it s an interface property event
chore styles add radiobutton defaultstylekey
remove unused parameter
fix remove unused null check co authored by cody robibero cody robibe ro
update product name to jellyfin server
add clarifying comments
chore remove treeview unit tests from windows
update newcommand cs
force attacks always go through regardless of what is beneath them
add support of animation on transformgroup for android
these method calls are impure
re enabled jpegsystemdrawing
fix drag select crashing
prevent div zero exception getsufficientcapitalfororder
merge remote tracking branch upstream master into song select performance
repo git when things are utterly wrong abandon rebase
react to logging changes
make test culture independent
fix osubutton doesn t implement ifilterable
revert some redundant edits left behind after merge
observe exceptions on the task returned from the queue
add esc clear marked reference command binding
unhack weird neutral player getter
disable a gsharedvt test which depends on async functionality which no longer works
fix ffmpeg preview to show screenrecordfps
change bolgarian towords
variables apparently for iphone does not allow compiling in winrt
async open subtitles
add some more negative tests add a couple negative tests around auto property declarations
merge remote tracking branch upstream master into installer updater
some little size adjustments
remove need for pragma in bridge cs
force vs to treat all blazor cshtml components as inheriting from blazorcomponent so that helpers like onclick appear in intellisense
maint todo note
pass along correct cancellation token
remove test code
fix don t try to find episodes if parsing failed
this should be done in the stage instead of the playfield
make todo abpintegratedtest class namespace change
fix wrongly negated if statement
rename livetradingresulthandler json
fix some file size limits and a missing counter in onedrive bits transfer
apply binding redirect policy before loading an assembly into reflection only context
updated the source code comment
use defaultbrokeragemodel for paperbrokerage
remove eoln check from markuptestfile
simplify switch generation
do not run flow analysis on invalid code
put threads into background before starting
remove wrong overrides in requestchannelbase in discovery
add final newline
remove some redundant code on request from prabirshrestha sparkles
revert unchanged file
fix and add timespanwithprecisionandalternativecollectionformatter tests
add security attribute for some weird edge case
add author column tool tip
repo add todo
skip shutdowntestrun on linux too
remove debug code
make number of hours human readable
crash fix in formcommit after add new submodule
as with every new file use tabs why can t vs respect editorconfig when creating new files
update startup cs missed a config move
validate overloads should only be virtual if they accept a validationcontext
add createentryfromstream extension method
add test coverage
fix origins not being set correctly when reverting from reversed playfield
add redelivered flag to prevent failed algorithms circulating the cloud
update servicecontroller cs
use tuple name
fixes wrong palette for crushed death anim
allow auth status get requests by default
change xsd to only provide uniquetypes and remove generic type definitions
provide fallback to for device haskeys
remove some dead code
add typescript s language service guid into the settings map
it s no longer beta
be resilient to null paths names for documents projects in sqlitepersistence layer
skip tests in referencedecoderbenchmarks
add assertion the workspace type is what we anticipate
fix dumb name at bob s request
fix nre if package download fails or is corrupted
add test for value clearing via indexer
flip anchors and origins
use new header in the rankingsoverlay
remove test code from homecontroller class
fix context menu not closing when an item is clicked
activator change requires typeactivator to be singlton
don t set the scheduler in a test runner in xamarin forms
update registerservice cs
zero portfolio cashbook before getting balances from brokerage
allow customgameplayedwhilefarming to make use of appid and gamename of the currently farmed title
skip checked event while updating cbstartwithwindows checked
add proper null check
don t collapse sequential namespaces
improved playto profile samsung f series
remove references to displayname
add play resume menu to detail page
fix for dsa
fix audio image not extracting
don t change the animation when it s not your turn
submodules path fix in some situations
add back support for other method references in disassembler as it is now supported by srm
cr feedback no need to use visit to create a constant expression
obsolete mapper initialize
fix mgg mrj spinners
cannot test namevaluecollection for booleans
skip test on coreclr
fixes bug in setholdings with liquidate true
remember index to restore before file deletion
urihelper reduce visibility is now internal
gazelletracker fix artist parsing
fix macos grid inner control display text
do not log page not found exceptions
reduce nesting and add comments
add todo for ro handling
bug fixes reset when applying the yoyo effect creating a new recording without discarding it first
targetinvocationexception is converted properly in javascriptobjectrepository trycallmethod
simplify null check
appease the resharper gods
clean up document taiko health processor
move playlistsresultsscreen to correct namespace
disable a corlib test which fails on some test runners
fix multiple spaces
move listener call outside of event handler so that when listener subscribe to the event service sees the subscription
make sure we keep reasonphrase from existing response
editionfeaturevalueprovider should handle null principal
update networkissues cs
those should be errors
prevent wpftextview mousehover from getting raised if mouse is over search control
fix merge conflict
remove test code
fix build on not winrt
fix startup crash on mono
fix pressing escape too fast causing multiple exit attempts at song select
test ignore a test case if executed on the main thread
simplify expand number check
add constructor that takes a text argument
add a textview role to enable current line highlighter used by log repl
include water and waters as uncountable for english
apply further framework changes
increase wait time to help test on ci
document default connecttimeout publicly
remove unnecessary local store
hide some internal razorpage members from docs they are used by razorgenerator and not intended for others
fix waveoverlaycontainer always being visible
changed parens changed parens
restore platform compensation
increase httpclient timeout
remove local chat echo remote should echo everything now please
merge branch master of tig migueldeicaza gui cs
fix scores header is visible even if no scores are loaded
fix nre in binding path parsing
dont clear textbox selection when right clicking
fix broken tests
fix udp server messages
skip workspacecoordinatortests filefromsameprojecttogethertest
use new captureproceedinfo feature of castle core
fix don t use extended episode number as release group
tidied up threading for wp
fix the drifting away of the gametime
remove old key when migrating to the new key
adjust reference for test
change metadata redirect path
small condition fix in readerwriterlockslim avoid an extra loop pass
wasm enable test_0_arm64_dyncall_vtypebyrefonstack
update csharpuserangeoperatordiagnosticanalyzer cs
show failed message on ffmpeg download error
more changes on default installation enabled externalauthenticationsettings autoregisterenabled setting
pass the correct owner to the messagebox show
ios now supports pvr compressed textures
ensure hotkey actions are executable in handler
statusicon change wording of sync statuses
no more box background
skip overlapping diagnostics
fix on linux clients without a screen
xbuild convert pbdonly to portable debug mode
fix grammer issue and more rewording co authored by bartłomiej dach dach bartlomiej gmail com
fix crash on startup when specifying no version overlay
update src features csharp portable codefixes addexplicitcast inheritancedistancecomparer cs co authored by cyrusnajmabadi cyrus najmabadi gmail com
ensure pageant is loaded to the tray
touched something i wasn t supposed to
convert to auto properties
remove unnecessary comment
no need for continue either
fix some doc strings and made the informational message userinformationexception
update csharpuserangeoperatordiagnosticanalyzer cs
add test for indexers indexers are not supported
reduce assert to guard in the outwards direction
fix modtimeramp not working
update message in jitoptimizationsvalidator
system drawing on image load from stream errors return null not an exception
only schedulewrites on the threadpool
force explicit conversion ref i on binarynumericinstruction except if operator is or
replaymetadata style fix
fix localization test failure in testreportingunsupporteddiagnostic
async not needed anymore
merge remote tracking branch upstream master into general fixes
fix nancybootstrapperbasefixture cs
add test for selectmany with identity lambda
decompile comimport attribute
return room model when joining
better error handling
add responses to multiplayerscore
remove perf testing code
bugfix for the home location
remove unnecessary code
add exception message
suppress fxcop warning re resource in signalr exe
reacting to lockfile changes in xre
update comments for the featurechecker property of applicationservice class
c p fail
fix one off issues
get request won t work with body so i remove the test breaking change
disable test that requires some design changes to fix
avoid wrapping analyzerconfigoptions in a dictionary
disabled moc moo order test temporarily
no need for partial
move column after message column
add descriptions to the fields in sellable which lack them
fix a crash in the move activity
assert when emitting a box il opcode it is not for a reflike type
fix missing base method call in carryable
itemscontrol fix null argument in onitemschanged
apply fix to all systems
add serializedname to enum flags
typo in advanced application settings noticed this entirely on a fluke just doing my part to help great application
react to removal of directivetokendescriptor value
maint comment about extension including dot
merge branch master into compiled bindings inherited interfaces
made announceonbuild implement inotifybuildcomplete
fix missile projectile crashing with nre if it has no image
make sure user has access to cipher
intro update statusicon menu after succesful clone
source code typo fix
use while loops instead
facades updated netstandard facade
unfail game start
dynamicproxyignoretypes should care about thread safety
fix my bug in encode_workswithdiscontiguousbuffers
sealing some classes
query for input manager once
remove some ifs
repo don t dispose common listeners watchers
only refund captured charges
changed to abstrart
potential bugfix in removeprojectfromsolutionstep
update nugetpackagetolocalreferenceconverter cs
fix test failure due to merge conflict
fix incorrect relativechildsize in catch
make showmousescrolldropdown static it needs nothing from the instance
refactor unhandled exception handling loop as per review
update emby server implementations httpserver httplistenerhost cs from review co authored by cody robibero cody robibe ro
parenthesized is patterns are simplifyable
create slider stackedpositionat method
add flags to replication change options
guard against double dispose
show split counts on split image button
changed entitylink exception to be a warning
uriparser is now able to print empty user info
other misc fixes in platform
move control static constructor statements to more accurate layers
move file move route use datetime in query
skipped failing tests for now
remove no longer implemented websocket feature from collection
only create dictionary when analysis is enabled
torrentech add compatibility for all themes
fix merge error
add highlight palette
prevent non combatant players from receiving units from crate pickups
shake in all cases
restore red lines for harvester targets
flush on savestate
merge branch apply to track into adjustable rate mods
don t raise global options changed if nothing is changing
fix a wrongly formatted string
implement specializingdecompilertypesystem resolveasmember
adjust syntax errors check
rework declaringtype filter
use local createfont method
add experimental flag
apply mod score multipliers
don t block the event log on initial startup syncs
add missing collapseboxsetitems parm
prevent crash when actor is null
not used anymore
org name for mail url
lazy load scripts
disable a not working delegate test
picture box label needs to be realigned on dpi change
further changes on the previous commit updated source code comments
textbox keep caret index when clearing selection
custom plans with expiration are not trials
show d rank badge on accuracy circle
use hashset to prevent duplicate consolidators
add back forgotten score store to valid importers
swapping silly uk english
add constructor overloads to compositeasyncdisposable
move wedgecontainer transition to onentering
remove autofac s extensibleactioninvoker during initial dependency registration injectactioninvoker breaks checkout completed route
fix regression in trianglespiece
changed js content compression settings allow http and https protocol prefixes in uri based attributes
remove other equals object calls
fix default content buffer max size
prevent variable frame rate gif with desktop duplication
update sync menus
use correct configuration object
update entitydynamicparametervaluemanagerextensions cs
implement more review suggestions
fix algorithm download tests
fix the mispelling
crate actions can now define required prerequisites
fixes bib on weap
remove test try catch
address cr comments
fix build error
fix merge memory leak
fix inverted order message for option assignment expiration
torrentnetwork improve passkey handling
remove redundant namespace qualifier
fix incorrect end time calculation in songprogressgraph
fix bad argument for non gles platforms
add slight easing
ui remove debug lines
focusing selected node after history navigation
moving event was remove now calls notifymoveorresizestarted directly
don t bother with teh fuzzy matching check if the matcher was created with fuzzy matching not supported
re adds ignore to brokerage integration tests
have proper cleanup of the enumerator
add includeregistrationservice option when registering the authfeature
add includetrailers param for similar movies
clear graphicsdevice _enabledvertexattributes so they get re enabled
lift the input output match restriction as we now properly loop over just the file list
remove duplicated code from settingcontroller
fix typo in sdl gl bindings
fix a typo in previous commit
corlib harden a gc test against possible false pinnig
test ignore when_readdatetime_minvalue the test may fail if the timezone offset is negative
remove duplicate gethandler wrapping gethandlerfactory invokes gethandler which already wraps the provided factory with gethandler
adjust cipher length validators
add rectangles combine extension
address more comments
implemented graphicsdevicemanager begindraw enddraw
seek to start time after placement not end
remove builtincultureconventions pathculture by default
autohide false by default for drawing image effects
itopicservice registration was listed twice in dependencyregistrar class
nuke usings on xs
set deployed application root folder as working directory
fix incorrectly build tests
remove extra convert method
remove duplicate nullablecontextoptions property
reduce closure allocations in coderefactoringservice
fix ci issues
also don t leave a husk somewhere when loaded as cargo
add hover and grab cursor support to all image editor controls
fix comment formatting
statusicon remove ambiguous reference on ubuntu
a bit of cleanup
add missing param
allow multiple usages of knownassembly and knowntype attributes
fix problem with the duplicate images on the db table
fix potential double disposal of player if playerloader is finalised
skip checking thumbprint of file in osx
update docs again missed keysplinetypeconverter
more clipboard macos fixes
fix null reference if hit lighting is disabled
move cef isinitialized into constructor fail fast rather than waiting until load is called
fix assert in diagnosticanalyzerdriver to account for suppressed analyzers which are not passed to compilationwithanalyzers
fix wrong check
set culture to localizedictionary instance workaround for elementhost bug
move common width definition to base class
respond to pr comments
after import change language
remove blocking calls from tests
fix model labels
add comment to sequenceequal
update omdb provider name
don t show snap areas on freehand region
supressed xml comment related warnings for tinyioc httputiliy and simplejson
fix incorrect osudifficultyhitobject instantiation
fix off by one colour retrieval
beautify the sqlite project code
add back billing settings
remove useless clock assignment
fix the add integration test
respond to pr feedback
avoid multiple access to selectedbeatmap during y position computation
fix sample and instructions
fix build error
revert unintended change
fix q press crash
windowsbase fix potential string overflow bug when getting zip entry file length
two small typo fixes
workaround for legacy nunit
add capture test
fix usings order
chore styles fix compile error when resourcedictionary xaml files have absolute paths
add comments for coarsenolookaheadbiasalgorithm
move selection blueprint to correct namespace
remove unnecessary check
update code actions baseline for csharpcodeactions integration test
fix convert to tuple analyzer
don t run member verification twice on nested types
fix references to pp form and add different options
fix issue with command line mode
improve discarding non ascii symbols
add an overload to createcollection that takes a selector
add offset to column tag helper
wrap job creation process with distributed lock
throw not implemented exceptions
add missing licence header
optimizing layouts of contains
fix setting bit
cache selector tostring was actually add up to a noticeable amount of time in large layouts
appease codefactor and appveyor
remove nonexistant function call
ignore redisservereventtests in ci for now
increase delay for window hiding on full screen screenshot
change test work be inside type declaration
add test for stackalloc of function pointers
revert left over type change in skinconfiguration
add combinewith that can combine objects ints etc saving casting at callsites
handle empty regex
only add custom error messages to statusdescription
update tests namespace
fix right click context menus appearing in incorrect locations
adjust padding on drawabletopscore
fix storagefolder adjust createfileasync string behavior
update tests to only bind to the loopback ip address prevents dozens of firewall prompts from appearing while running tests
fix async context settings
change isprofile to use readonlyspan byte
implemented pushopacity for cairo
add spike limit allowance
formatting fix co authored by pavlukivan pavlukivan outlook com
reword comment slightly
add missing using
per comment simplify code in setter
change to use stringbreaker api
clean up usings
remove unused write decimal method on endianbinarywriter
make fields readonly
initialize bp error tooltip
update viewbagtests cs
tests adjust ui test sample control lookup
socket move duplicateandclose related methods to their region
add profile and level to progressive streaming
update src tools source compilergeneratortools source csharpsyntaxgenerator abstractfilewriter cs
fix distance snapping grid not updating on scroll
remove output spam from a unittest
disable menu close on click
off by one on colors choice
check terrain cost when testing pathable cells
remove watermark text from image and text drawing effects
skipping mono tests on both clr and coreclr
add createdat to meta to capture when subscription was created
minor fix in description for lockoutenabled property
change box to editorplayfieldborder
fix order queuing on passenger
better message for integration tests
remove duplicate code after merge
log refresh the event log when there are changes
work around a race condition between server join and auth validation
add failing unit test
only watch local user to prevent conflict between testers
add a constructor for the single section case
remove returns xmldoc
remove unnecessary partial keyword
update mac startup
update link to the docs
set font in music controller
flsuh always explicit
fix gamepad back emulation
fix default channels not being joined
made assembly catalog methods virtual
remove mouse wheel disabled checkbox from visual settings gameplay menu
minor schedule tasks validation
remove unneccessary string
load pdf logo image from content images directory not from images
add useful link to digestauth
remove kill death count in player health
add virtual empty event handlers to abpwebapplication
merge branch master into js wu quantizer
perform notification date check in utc
fix synchronizeitems add duplicate items
abp select tag helper bug fix
revert change hang to blocking
restore no arg exponentialsliderwidget ctor this is required for dynamic widget creation
cardigann add support for imdb movie search
adjust a couple of flag scales to match previous display size
changed gpg tab setting to show by default
remove dryjob from introbasic
fix odd error with simulated latency trying to return null in release
fix welcome sound being played twice
change to getruntimeproperty
catch force locale error
update generic cs
reviews add unit test and improve exception message
add mention of notelock in xmldoc potentially easier to find class
only say select target when pushing chrono not canceling it
react to caching api updates
change name of telemetry
load playbutton preview asynchronously
use full name
use appsettings font
setup select on my own server plugin by default
fix spelling mistake
jobject fox for ciphers push
remove warningsareerrors when compiling shaders remove unnessasary at least for now if windows
oops nodelay s
fix failed tests
add missing throw keyword
skip complex type property for property changes
fix a crash issue with temporary backups
fix serialization crash
change the words
fix netstandard build for nullable annotations signed off by dotnet bot dotnet bot microsoft com
ensure laysterrain doesn t access cells outside the map
updated error message
update updatechecker cs update updateurl
don t evaluate twice
add missing abpmodule to cms kit module
update device cs
fix possible argumentnullexception in contenttypereadermanager contenttypereader targettype can be null e g for externalreferencereader
fix workspace references in liveshare unit tests
update sessioncontroller cs
remove some todos
refine policyregistry intellisense
start log scope only if logging is enabled
merge branch master into fix circularprogress glow
remove some extra dispose code
fix pi and e filter bindings
error page we cannot have ul in p tag
fixes force fire not working on pillboxes
fix bug in vb project export
fix not posting to the correct channel
use thread instead of longrunning for api
remove line break
only challenge if not already accepted
fix crash in td s server creation dialog
add opengl graphicsadapter isprofilesupported
change method to property
move comment to apply to correct code
resolved nullref error with setparent
fix taiko hit target alpha on legacy skins
optimized async flush in serverstream
add async tokens
don t bother with an exception that will never happen wasn t being caught anyways
wrong method was used in productservice
parsequery is now skipped if scheme does not support query
remove unused using statement
make sure caret is visible edit value control
fix reg adjust runtimetest for leaks validation condition for ios
no need to cache checksum here
fix for wrong line endings in pipeline tool output
fix compilation with old compiler versions
don t save date if is the default min value avoid empty dates to be saved
update with framework changes
fix proxy tests for localized environments
throw a better exception internal only
update typestyle cs
add another automationelement helper
don t lock scroll position of sectionscontainer header elements on negative scroll feels better in all cases
better error message for fix size collections
don t run oauth test by default
call only measure and invalidatearrange metroprogressbar
completed gtk window implementation
use correct annotation to detect custom operators
return from skirmish lobby to singleplayer menu instead of main menu
add getorderbyid method to transaction manager
remove unnecessary room id from leave room request
use bindto instead of taking the game s beatmap bindable
scopecontext fix spelling in xml documentation for pushproperties and pushoperationproperties
fix ff resolution in filesystemdatafeed addsubscription
fix merge conflicts
maint variable naming
add helper to shorten logging syntax
ctrl o for options
merge branch master of https github com quantconnect lean
gecko make ioncannon and nukelaunch effects public
split ignore into hit miss
add new episode number unit test
sorted scenarios and cateogries
fix replace error
use correct mediainfo method
fix a spelling error
add keybord input filter to only register first key press
updated data folder selection
also remove tokencookie on auth logout
disable button while update check is in progress
set security level to full
attached properties can t use nameof
do not collect remindertablegrain when shutting down silo created noise in tests
replace o n m with o n
remove unused function
write a better comment
fix ambiguous id in movies sql call
update filtercontrol using new accentcolour definition
unsafeinitialize is no need
add getruntimeconfig api to allow overriding config in multi tenant apps
fix load config check if server node is null
renamed filter method
split up add of local store from the ctor
using typereflectioncodegenerator in unitycodegenerator
fix the build
react on review
fix result screen not showing name of mapper
support not finding a factory
start marking cut copy paste methods obsolete will soon be accessible at the frame level
fix floating text showing up beneath shroud
set content type of publickey to xml
bring back original logic of test still trying to figure out what s going on
use new isobsolete method in documentsymbolshandler getdocumentsymbolasync
update plex movie libraries instead of series
add link to explanatory comment
add an inflate overload
fix crash on trying to retrieve mods from unavailable ruleset
update playtocontroller cs
updated tests url
avoid downcast in invokeworkitem
add a lua api method to randomize a collection
export otion serializers for tests
fix capitalization of asp netcore check
add missing acronyms in testcasedrawings
create plinq query tasks attached to their eventual task parent by default
fix the floating text for bounties of actors not being inworld showing up
un fp building cs power stuff
implemented another complex cue variant
cardigann don t use empty banner urls
merge remote tracking branch origin dev into dev
shadow should be disposed in textrenderer according sharpdx docs http sharpdx org documentation api t sharpdx icallbackable
logging for travis these tests pass locally but are failing on travis and i m gonna find out why
prevent stackoverflow when recursively disposing ioc
add additional blank test
fix a bug in he mgcb intermediate output folder creation
add the handler back for the add item tree context menu
add back missing using statements on psm
yeah cinema mod isn t going to work
double check connection handshake mainly for mono where the earlier flag isn t available
add confirmation for package overwrite
warn faster for slow requests
fix comment spelling error
merge branch mergemarkers of https github com cyrusnajmabadi roslyn into mergemarkers
make gitignore creation more portable
fix wrong english
fix non deterministic enumeration in subscriptioncollection
call getsyntaxrootasync once outside loop
another fix misspelling of dependencies
remove volume reduction on preview it doubles with global reduction
allow importing effect when list is empty
fix unit test compilation
remove some unused using directives
moving keepbenchmarkfiles merge logic to add method
add trait documentation to infiltrateforcash
plinq enable first and firstordefault unit tests
fix pcl build error no foreach
add netmodule to treenode if the loadedassembly instance represents a netmodule
fix build error
set title in aboutpage
git add some more ms office exclude rules
add comment about ignoring empty files in projectassetsreader
fix trending bug with escaped languages
simplify base type interface check
fix docs on socketnodeinstance namespace was incorrect
fix typo of point parse
update failed lambdaexpression test the failure is caused by a bug
add doc comments for projectstaralias related method in selectexpression
make notsupportedexception partial to allow xi to add an helper method
add unneeded whitespace
decompile volatile modifier
update src scripting coretestutilities objectformatterfixtures mockdesktopspinlock cs
changed naming of jsonconverterattribute converterargs to converterparameters for consistency
remove commented code
fetcher git ignore mercurial and bazaar databases
noticed a small error and fix it
wording fix case
only apply delayed bindings to toplevels on load otherwise e g styleresources for usercontrols aren t found
fix null baseurl no null check prefixes uri with forward slash and turns httpwebrequest into filewebrequest
don t open profile if it s autoplay
remove all updatesource except textbox all others have the defaultupdatesourcetrigger not on lostfocus and therefore no problem
commandlineoptions to abpcommandlineoptions
genproj add support for mcs getresourcestrings switch
use new getinitializeduint32array method
fix cors tests
remove validkeys now in drawabletaikohitobject
fix invalidcastexception in the jobhelper getqueue method
don t batch imports for now
marked capability providers with exclude from code coverage
bom disabled by default on ws target
add convert for test
threadlocalstoragehelper netstandard should only allocate when required
use netcore for commenting out tests
temp patch fix for png encoding i have no idea what is wrong
deleting some commented out code
ensure that calling compilationoptions with doesn t lose the syntaxtreeoptionsprovider
made debug text yellow and background the default color
add new constructor with iwindowimpl parameter
fix nre set soapmessage in the right place
fixes dpad not being processed in isbuttondown up
include statusbar height on y calculation
merge branch release into dev
add a missing configureawait call caught by analyzer
merge remote tracking branch jbialobr mergebyhash
react to stringsegment change
typesystemastbuilder do not create references to system mathf if pi or e are not defined
add a removenode miniyamlnode method to updateutils
update as per suggestions
remove console writeline
the texturestore in country cs can noit be null
skip palette bytes if image type indicates its no palette image
remove call since value is cached
revert internal on method
refuse to start the game unless required slots are full
fix typo in totpsecuritystampbasedtokenprovider docs
window s default behaviour double click on title bar will restore minimized window
fix failing test in addimportstests
update authdtos cs
adjust cache policy
git fetcher actually trigger progress event
add a test for slicing on this
line can not be null in osulegacydecoder cs
tests the test crashes the runtime
restore overlay mode for splitview in minimal mode
remove unused variable
made lengthadapter check for implementation instead of interface
add check to see if backends report the same file twice
renamed a variable
toggle pause in editor with spacebar
reimplement throws because android s nunit doesn t do it the same
gtk pipeline fix null reference exception
make player autoplay can be set
add test that explicitly blocks for a response to make sure the client does not deadlock
add getall method to isettingdefinitioncontext its implementation class already has the implementation method for getall
use reachable host
add setting for showing stash count
reacting to roslyn changes
fix game locking up on crash
remove sliderball s input override
throw the correct exception if receive is cancelled
fix bug in introduceextensionmethods
the option is not per language
changed the default date format of moment js library
respect the x http method override header when choosing a handler
update accesscontroller to use the setinternalauthorizationid extension
add missing texture reload for directx
remove custom log path
remove unused using statement
fix build break due to model binding changes
remove duplicate test
fix bindable not working under non debug due to weak references
add protectiontokenparameter for mutual nego patch by alexander riman
call the mouse event
remove iconfigureservices from grpcfeature grpc doesn t support modularstartup
corlib test check for israel standard time zone on windows
add comments for segment of grid
remove error message duplication
fix wait for runtime test control
handle additional idlewarning state
reenable drop shadow hopefully the last fix for this
doh do proper osx detection
changed preparerequest from internal to protected internal virtual
fix addstagetests on windows
add icrud interfaces
remove duplicate configureswaggerservices method
cloudflare increase retries
adjust xmldoc slightly
graphicsdevicemanager preferreddepthstencilformat now properly defaults to depthformat depth24
give hitobject some hitwindows
ajaxresponse fix made return type ajaxresponse since generic version is not supported by web api
use switch expression
add log message when homepage is disabled
remove unused usings
hardcode datepicker month day to prevent unrepresentable dates
fix build porting problem for about window
add name id field to withspritebody
fix wrong license header
remove unnecessary usings
fix an issue where running a backup job would reset the log level
microsoft build engine implement converttoitaskitemarray create a taskitem containing the result of the invocation
rethrow exception after logging this ensures that the backup properly fails
maint ensure no null on enumerable iteration
merge branch master of github com sharex sharex
fix clipping transform issue when saving and restoring contexts out of order
fix bug i introduced into tests
formverify also display creation time of the blob objects
quickfix for transactions timing out and not committing when preparingenlistment done is called instead of preparingenlistment prepared
make success rate percent move with the bar
add comments to explain empty methods
fix an issue with uwp clipping
add clientname to output for timeout messages
fix nvorbis instance not running on game end
change the sourcerouter configuration from local to remote
cloaked units not on radar untested but is simple enough that it should work
update applicationhost cs
add missing license header
show that s you in the e mail position
shorten fade duration to make fade out snappier
unwind some linq in a relatively hot path
merge branch master of github com servicestack servicestack
fix overlapping shadows
beatmapleaderboard fix local scores not being sorted
move httpcontextcancellationtokenprovider into abp web
fix flyout autoclose sample
revert some more changes
improve logging in test user profile service
fix some code
fix exception in lastfmbaseprovider cs
merge branch master into settings flexibility
minor refactor renamed to xmldocument
sync primary solution more frequently
update comment to explain the reasoning behind showing sig help
wrap table storage conflict error into inconsistentstateexception
fix multiplayer create button appearing in match song selection
dpi scale formstash splitter position
fix unit tests properly
catching exceptions and in unitycodegenerator and clearing progress bar
update diagnosticanalyzerdriverallinone per syntax changes
remove useless class name prefix
set versionname null rtm
assign uninitialized out parameter
point angle interface to avalonias custom built optimised angle
json serializer now recognizes any scriptignore attribute regardless of what namespace it came from
remove unused usage directive
update sqlite sync mode
make symbol string overrides as obsolete
change dependencyresolver method to property
update httpresponseextensions cs
register for hardlinkcreated on windows
renamed sessionchangedeventhandler to puttysessionchangedeventhandler
respond to pr feedback
fix wrong gui color for contexts with retained entities
fix some possible null reference exceptions
allow player to not pause on focus loss
remove hard coded info about system valuetuple now that we re add it to the nuget index
folder can t be created previous to applymetadata
fix materialcontent gettexture to return null when a texture doesn t exist
changed defaultacceptheadercoercionconventions to use acceptheadercoercionconventions not staticconventions
revert allow valid https requests in net core
assign to field and move to load complete
ireactivelist t already includes ilist t
allow inlining into object collection initializers
remove editor functionality from virtualbeatmaptrack shrug
reseal image tpixel
socket move sendpackets related methods to their region
spelling public identifiers in src microsoft wsman management
fix code style issues
fix warninglevel comment
merge branch release into dev
add method to find resource from strings in code
fix test scaffolding
add issystemendpoint property and treat system endpoints as non service
remove rreduntant initialization
fix mac build
formpull fix spelling mistake recommeded recommended
copy product functionality should also copy a custom value of specification attributes
don t acquire page lock during counter aggregation
fix importing of private types under dynamic context
remove parenthesis from positioncursor
reorder methods alphabetically
don t log disk space related io errors
test for not inferring inaccessible locals
mono debugger soft add typemirror isnested
events were not checking for null so were throwing an exception
better version of the previous two commits async exception handling
seal the criticalsorthandle class signed off by dotnet bot dotnet bot microsoft com
rename parameter name in requesturibuilder
fixes default category of qbittorent client
use checkexists function
use bitconverter islittleendian directly this allows the jit can optimize away the unused branch
remove extra spae from worldpay plugin
use environment newline
updated version for versioncheck
improve valuetypecall using test case
avoid null ref in clearalldiagnosticsforproject
disable unused endpoints on identity server
fix system io compression filesystem zipfile createfromdirectory overload to use compressionlevel
ensure we can set double properties with ints
clean up code
use another frame if exception frame is null
move call to trysetcanceled to avoid unnecessary finally block
fix index of searchcombobox
expire placeholder text
load player based testcases asynchronously
attach canexecute to bound commands
do nothing if control was disposed
update src workspaces coretest options optionkeytests cs
maybe t now calls t tostring
system core enable disabled test
expando implements idictionary string object
fix a property name mistake in gradientbrush
name tuple items
enforce order modifiers
update subtitle quotes
fix build error
load stack for reverse using enumerable taking ctor
update semanticclassificationbuffertaggerprovider cs
minor source code refactoring
fix getboundcopy on igamebeatmap not working
update template for the objectextending change
use getrequiredservice instead
blobstoringdatabasetestbase should not derive from blobcontainer_tests
add fix all test
update csharptypeinferenceservice typeinferrer cs defaultstatement defaultexpression
add test for missing binding behavior
increase silent period
fix blank line
call ihistoryprovider initialize from lean engine
change fo fix windows call to gl getstring
remove commented out code
load psreadline by correct name linux is case sensitive so the modules are too
fix ss dll config
fix resolvebyname extension so it can never return null
add test covering host retry after returning to spectator screen
only return fallback cached response for get requests
fix leak when an error message is returned from sqlite3_exec
increase the number of bins in the timing distribution
remove unused code
log every fill event in live mode
override basic equality function too
don t use proportional resizing for side nodes
fix a bug in porting
system windows forms fixes test build failure
remove double schedule
xbuild evaluate import with msbuildthisfile properties set
update with missing icons
default to allies for neutral in ra gdi in cnc
fix issue with comment
add validforresume check
update filter label to better express what is done under the hood
don t advertise we support hover handled by liveshare
the arrange mod is not ranked i think
fix for morethantv and some ssl issues
don t start branch deletion with empty selection
make createscaledbitmap an instance method
fix parsing of some daily episodes
remove unnecessary optional parameter
for fields too
avoid boxing in geteventhandlers
fix nre when a client closes connection
update mediastream cs
put back actual save of module target
replace nested loop with early return
fix retry timeout when running under debugger
changed start position to manual
sometimes static members aren t initialized properly for some unknown reason
disable warning about unused class
fix wasm does not have multi threads well well well
update invalidtoken error message
fix videoaudio conditions in streambuilder
show messages received from return values in httpbasedtransports
fix a bug in the graph dominance calculation
polishtracker remove unused cats
rollback countersaggregator to its initial state
current song dispose was throwing exception changed to handle disposing the song object more gracefully
fix testcasetaikoplayfield not correctly setting strong hit objects
fix linux keyboard minus
fix class name
avoid unobserved task exception when logging a message rejection without an exception
make iinsightmanager inherit from idisposable
remove unnecessary code
beatmapleaderboard use orderbydescending instead of negation suggested by peppy
log the catched exception when a client is dropped because order dispatching failed for debugging
add workaround for dnr patch bug
fix some path issues
don t invoke parallel for if they are no handlers for a signal
updated applicationdeployer to conditionally delete published application folder
add comment s
search ui make sure alt r a don t work in ui unless replace ui is shown
fix hopefully metadatatypeattribute not being found on some platforms
implement isempty directly
update logging cs
add tests for virtual and abstract properties
make showslotdropdown static
fix cursor add
fix captura shot failing for full screen screenshot
add new custom api s for executing raw commands
remove alwayspresent not actually required
add assemblyproduct attribute
update viewwindowmanager cs
fix the parameter bug
add summaries for istatehandler and istatechangeprocess
add using statement for extension methods
make hasexitedsafe private
remove obsolete test
add test that validates behaviors
ime patch for commit message editbox
fix filetarget thread was being aborted fix filetarget thread was being aborted thread sleep can throw a threadabortexception
merge remote tracking branch upstream develop into develop
only drain health in non break times
allow di d onscreendisplay to be null
throw on invalid argument passed to diagnosticreporter report
improve erroe message
fix potential crash if disposing a drawablestoryboardsample twice
fix missing attributes
remove overzealous check
xml out of range errors should be swallowed when doing string datetime conversion
allow clicking notification to open screenshot folder
updated wcf host to support the new response model of action stream
hide top score at every leaderboard change
reacting to verbose rename
add unit test
fix livetradingrealtimehandler pack up creating the potential for a nullreferenceexception
old geared key
do nothing if the new selection is equal to the old one
modify some words
decrypt encrypted sl resources
introduce another null check
remove unused usings
update servicehandler cs
last directive error
changed onedrive scope
add missing blockingpopandpushitembetweenlists to typed interface
merge branch master into storyboard improvements
update completeddownloadservice cs
fix for the private setters in base classes issue
improve propertygrid dark theme colors
fix hex caret pos
fix merge errors
cleanup a bit
remove refreshing of session from real time handler
fix fxcop issues
log refresh progress at debug level
add more docs for threadagnostic usage
managedweakreference use objectdisposedexception type makes errors easier to interpret
made multi threading test skipped
enabled dither again
merge remote tracking branch sharwell fix build into symbolequivstackoverflow
no need to popup the settings dialog if we can resolve all settings automatically
actually add the background
restore previous behaviour of vertexbuffer setdata on psm platform
should finally fix exe icon and resulting build error
fix json line ending trimming
updated assembly version
add overloaded ctor for subject less grantvalidationresult
add default password to logindialog and logindialogsettings
update sharppagesfeature cs
react to security api changes
throw httpexception when tvdb sends us crap data
fix msbuild langver test
bug fix when soundeffect was disposed but not stopped
resolved merge conflict
handle hover explicitly
remove unnecessary code
_itemschanging flag is not necessary anymore
only emit kps log in backtest mode
fix iron curtainable doesnt stack invulneffect effects anymore
rename offscreen in assemblyinfo
fix a crash in demolish
do not required write permission for the root directory some hosting companies do not allow it
extra ordinary bump due to api breaking change
typo fix in source code comments
fix registration lifetime
process catch the closing of handle
update abpidentitydomainmodule cs
make ctors protected
add missing enum thanks to nephrite
one more offset remove
fix verification query
update smarttokenformatterformatrangetests for top level statements
react to auth
list started games by default
made changes per comments
all traits can contribute to the synchash
handle case to add trailing separator back
formbrowsemenucommands make menucommands list singleton
remove unnecessary usings
don t run in code style
dispose of location if it doesn t match the serialized location
fix song select reverting to first difficulty of current group on entering
invert hide logic for readability
fix incorrect name of parameter being referred to
revert sanity check that building is entirely within map trees break it
we now clear the processor parameters if you try to change the processor
using import cell name to describe callees
emby metadata name changes
renamed the test
torrent syndikat add auto relogin
render multi line command as a single line command
fix a bug where double nan and chartvalues from min was causing double nan to not display correctly
fix missing speed multiplier split onto multiple lines
remove extra argument
cratespawner combine deliverbyaircraft and deliveryaircraft properties
format tournament bracket output
forgot to include a file in the previous commit
use mapper serialize method for dbref id
fix declarepublicapianalyzer the analyzer can report three different diagnostics but currently only returns two from the supporteddiagnostics property
add null check to log method
add m c to get correct file name
update generated view files
fix node debugging by catching warning
remove redundant qualifier
fix sqlserverdatabasecleaner script
don t bring hidden controls into view
fix null reference exception
unregister workspace on dispose
fix potential test fail case
remove method call
avoid watercheck crash in base builder
add isroot and isvirtualfolder to dtobaseitem
update a couple of xml doc comments
fix logger level
add missing xml comments code cleanup
don t print message since the code is now much faster
correctly updated menu item style
fix multi monitor menu location
update customwriterstartup cs fix a misleading comment in sample
fix parsing of negative wrange
log log actual file ran instead of string
fix texture2d setdata and dxt compressed textures on pss
fix cross os wcf incompatibility mono really struggles with this and can t respond to windows clients
enabled file history for right click menu on directories in tree view
update dynamicentityparameterstestbase cs fix multipleuniquekeyfoundexception exception on entity dynamic parameter tests
minor fixes on converttoframeworkalgorithm typo and unnused variable removal
oops fix compile error
only schedule prefetchstep when in attended mode and using a cache server
add a test with a local function
ensure max value in clamp is at least zero
disable select_collection_navigation_multi_part2 test
test effectiveviewport workaround alignment issue in native scrollviewer
transform screen mouse coordinates to local space coordinates
pull documentation from the associatedsymbol
add more comments
add code comments to the switch expression
fix missing array bounds check in shroud isexplored
found offending missing namespace fix for monomac
torrentsyndikat add global freeleech check
increasing precision of timestamps being logged
fix hardware mode switch change detection
implemented resume method for cue sounds
use palette indices that map to sane colors in both games
remove unnecessary trace property from messagebus
return true if both combocolours are null
merge branch master into download tracking component
merge branch master of github com mbdavid litedb
paper trading init error handler
discard unused value
remove uncalled method
drawing public fields of unsupported types
texture2d initialize texturedata array
fix definite assignment analysis bug
cull some unnecessary whitespace
actually consume replacesbackground
add methods for overriding route description provider
now getting the correct uid and gid
don t set a default reset provider
api doc fixes
make abpaspnetcoresignalrtestbase abstract
fix the scaleout configuration
add logger to abpmodule
fix video looping not propagating when set too early in initialisation
remove cyclical json serialization tests
make telemetry to only log when there are multiple files re analyzed
index write validation now uses max_index_key_length constant
fix a mis spelled word
remove ihaslocation lookup from target
let user know automatic reporting happened
allow testcategory at the class level this was not allowed in mstest but it is very handy
fix generate editor for datatype attributes
fix another font usage case
fix null ref in populate switch
new make authors more clear
merge remote tracking branch refs remotes ppy master into main menu song ticker
adds jsonignore attribute to finefundamental endtime
remove redundant code
fix nested hitobject judgements not being remove
give repeat points a size specification
fix options dropdown labels never being displayed
remove implicit operator
try fix psm build issues i d done these ones by hand
merge branch release into dev
invite remove port variable
reduce projectile search radius to match warheads
collectnamespacesfordecompilation add support for security declarations
temporary fix for participant count being nullable
remove blank line
add exportvaluetypes option to vb net
remove sleep from eventsourcestreamreaderfacts
some todos add remove
flip frame count check
don t try to create a xamlproject for a null hierarchy
feature address feedback use value tuple int string as the sorting key
don t validate unique oauth usernames by default
add a maximum length for slider ticks to be generated
merge branch jwtbeareroptionssummary
short circuit to avoid datetimeoffset parsing
increase sleep on test
use correct property type
fix missing plugins
don t use path combine for linux paths
merge branch fix follow circle of https github com thewildtree osu into fix follow circle
add frameloadstart to iwebbrowser and add to wpf webview
remove incorrect assert
use guid newguid instead of path getrandomfilename for random commit messages
don t render the idle overlay during selling
implement managementquery querylanguage
refactoring drawpathprocessor to use a const epsilon
rename variable used for mania lookup key storage
delete unused variables in highlightingtokenwriter
corlib fix objectpool to not incorrectly assume it was succeeding when its trial period ran out reported by greg young
populate onlineclient properties from httpcontext query
reset market hours database before quantbook tests
push rejected error message format has changed
add help file for the command line
remove unneeded wait for exit
diff disable context buttons when showing all the file because they are useless
adjust accuracy display to match stable
remove duplicate assemblyinfo
showing subsequent status forms for the same owner
trait instead of skip
update channel sorting
make extra sure the state is disconnected after calling disconnect even if disconnect winds up throwing
fix incorrect musiccontroller drag handling
add feedback link to messagebox
add test for verb that hasn t been overridden
fix up an oaph test
fix createindexoperation filter generation use csharphelper literal to correctly wrap the filter with
fix smoke being triggered by healing as well as damage
fix throw factory
support digit separators hexposition
updated interface comment
add contentheaders to apiresponse
nullappender add missing securitysafecritical
update page cs
work around harvester unloading activity queue issues
add missing comma
update src libraries system security cryptography encoding tests cbor cborwriter map cs co authored by jeremy barton jbarton microsoft com
changing back to use typedescriptor
do yet another attempt to make this work
use localredirect to ensure local url redirection
fix test to have correct background color even in release builds
don t chunk websocket keep alive responses
avoid double type check
disable https in kestrel if certificate is null
fix stupid bug in prev and in release
print server status messages to the console
reduce debug output on parsedatetimegolang formatexceptions
chore ios adjust touchesmanager
sometimes i wonder if i ll forget to breathe sometime
add overloads to responsestatus class
maint need system console for debug
simplify math by making direction a float
update usercontroller cs
fixes terrainrenderer draw causing a crash because of negative firstrow value
refactor extract setcurrentculture method in abpwebapplication
remove reduntant comment
more test for rsa oid from friendlyname
add as a concatenation operator for rediskey to prevent implicit conversions back and fore
component check now tests for the correct rdp version
add resolution deserialization unit test
convert to autoproperty
hide poorly designed icommand methods
add data normalization settings in option algos
reacting to hosting rename
remove useless usings
saw a swizzle bug
shifted logging for server version
mono debugger soft add flagsattrbite where appropriate
don t recommend this in static local functions
baseindexer include original stack trace in onparseerror exceptions
fix possible nullrefs
don t show region capture tips by default
buildfix for platforms using sound
further changes on seo friendly urls with multiple languages enabled setting such urls shouldn t work when disabled
add unitofworkmanager to applicationservice
remove warning suppressions
update getservice to getrequiredservice
features ifeaturedefinitioncontext cs documentation changes
spelling locals in src system management automation utils
explicitely attach loop underlying tasks to their eventual parent
allow to use gpu with deferred renderer
add new feature servicediscovery enum
tweaked ifdef indentation
linker we need to mark nested types even if the declaring type isn t marked
add helper method initializedictionary
add score panel test
handle empty fields result
merge branch master of github com sharex sharex
facades add more comtypes to system runtime interopservices
limit readarray to only reading up to the requested number of elements
fix rectangle center code to return the center of the rectangle from the x and y origin
renamed point data property
more nomojo refactor
change logohref to refresh the swagger ui page
update entityhelper cs
use resolved instead
remove code duplication
merge remote tracking branch upstream master into fix hidden slider heads
use the previous conditional instead
remove irepository todos
add support for robohash no image service
merge branch master of https github com mediabrowser mediabrowser
merge branch master into dev
remove thread sleep and tidy up new method
remove unused parameter from algorithmmanager stream
fix broken build
rip reactiveui mobile namespace
if shift hold when creating first point then create two points that way polygon mode can create line
only azure ban on non self host
add basic test
merge branch master of github com duplicati duplicati
switch to looking up via uiviewcontroller so it s easier to use the routeruinavigationviewcontroller
update myprojectnamedomainobjectextensions cs
fix casing of local variable
implemented soundbank playcue string audiolistener audioemitter
repo git empty logs make me sad always show something when no changes this month
system timezoneinfo fix finding system time zone when etc localtime has a relative path
use a local function
test that generated globalsuppressions cs has no trailing whitespace
dampen saturated opacities
fix avistaz exception
ffmpeg simple gif encoding quality not implemented no measures taken to prevent use with audio
add missing imports
further optimize new web browser
cleaned up syntax
add entry point related errors as build only errors
add test for nested captures
save all plugins settings on settings menu close
reacting to ichangetoken changes
fix playersettingsoverlay not reappearing when hidden
fix some unit test
disable masking optimisations for scrolling hit objects lifetime is very tightly controlled here so all should be okay
update ischeckoutwithfilepaths for switch and restore signed off by derrick stolee dstolee microsoft com
fix samples typo
update regression stats for weeklyuniverseselectionregressionalgorithm
ensure idempotency of dispose
avoid duplicate exception allocations on failure path
fix usings sorting
wrap blob storage conflict error into inconsistentstateexception
s r serialization fix member indexing on reading deserialized members
add tuner setup to startup wizard
better life time ends
react to identity changes
update crossprocessmutexbasedsemaphore cs
glimpse web debugging tool didn t work with nopcommerce fix by a minor change
add forwarded headers to enable nginx hosting
remove testing code
rename resolverandget to getasync
remove incorrect test case
introduce another null check
fix directory suggestion when using browse button for source repository
add meta to list of reserved chart names
implement deathtypes support to ownerlostaction
add components folder to the allowedextrawebcontentfolders
fix unit tests
implement creation of mods
changed the imageresponse to use getfilepath method from genericfileresponse hostingenvironment mappath does not work in self hosted wcf services
fix drum rolls not being correctly converted to hitcircles
fix bug with anychangeinexpressionlisttriggersupdate test backing field must be public in silverlight using raiseandsetifchanged
remove extra line
output the total time for everything including fetching compilations
monodoc monotouch monomac export should be at same boost level than type member
react to trygetproject changes
add some whitespace
add comment on trygetvalue
fix android managed load unload traversal for missing actions
remove unnecessary spaces
fix potential nullref
react to httpabstractions changes
htmlencode raw url in htmlformat snapshot
add mono tweaks
typo nullablee nullable
update detail page
rename to blobcontainerfactoryextensions
remove quad rendering support
updated textureimporter with all of the file types that freeimage supports
replace loaded check with better variation
ignore keep alive when nopcommerce is not installed
make sure the contenttemplate test reference ui element by name instead of content
load user logging config file into application configuration
add signalrrealtimenotifier to notifiers
stop lying in the comments
migrate padding into the scroll content to not cut off effects
remove extra newlines
test casting to generic interface gives fully specified reference
add pomf uploader
inline and simplify local vars
merge branch master into improvebuttonstyle
make an easier override of overridehandlersfortesting
don t email me when new versions come out
merge branch master into tournament tools
add safety type check to drawablescrollingruleset
adjust loading container corner radius
fix jitoptimizationsvalidatorismandatorybydefault in debug mode
addressed spacing issues
remove unused packagepaths
use skinnablesprite for approach circle
don t offer attackmove og in cases for other players stuff
enable the mono security build for monodroid
test on the compiler optimized version of quicksort
disabled rediscache tests since they don t run if redis is not installed
use drawabledate tooltip for datecontainer
use a safer method of setting the clock
flexibility improved webrequest create wrapped into protected virtual http createwebrequest method
remove unused code
fix bug where you couldn t click on controls after scrolling
changed the backup file timestamp to use system time rather than utc
better handle startup when ladder cannot be read correctly
reacting to kestrel extensions
repo explicitly fetch from origin master
add difficulty rating to stardifficulty
fix typo than then
make einstein run out of the building when he appears
fix results screen crashing for beatmaps with no online id
testing wrench build
fix the macos build
never run subtree masking
throw argumentnullexception in cleanableweakcomhandletable if add is called with a null value
made getvirtualpath virtual for extensibility this method has always been virtual in mvc and web api
add documentation for tryupdateparentofupdatednode
prevent vs from crashing if vs got shutdown while build error reporting is in progress
fix android uno template for android now reacts to all configurationchanges
handle more vertex formats on playstation mobile
fix withattackanimation breaking withmoveanimation
add subscriptions to the topic on the fly during subscription
fix setting duplicate keys from auth header
fix constantbuffer setparameter
do not refresh if grid is disposed
fix issue where fire and forget delivery on the simplemessagestreamprovider was defaulting to true if fireandforgetdelivery attribute was unspecified
remove useless code
further changes on dropping sql server compact
ignore taskcanceledexception thrown when runtime has been closed
lower pool count
add sparse option for disabling and returning to a full enlistment
fix another confusing typo the file being add is the config of the last assembly read not the runtime name
fix an indentation style issue in transformsintoaircraft
remove unused taikoscoreprocessor constants
fix build error
expand docs to match
fix repair depot crash
corrected project due to previous changes vs don t recompile it
only set result in ios messagebox cancel if not already set
i think that cast is made elsewhere
workaround for unexpected exception
fix xml doc warning
typo relationl relational
change appbuilderextensions to use nancymiddleware midfunc midfunc is supported by katana
implemented equals and gethashcode in fontkey for skia typeface cache
i swear no more bumps tonight
changes to redis tests examples
add img patifile com
remove experimental wait calls
make internal method internal
fix multiple difficulty control points sharing the same time and thus confusing slider velocity calculations
previous method of handling missing enum values was not reliable
replace iuseridentity with claimsprincipal
add movie notification
fix highlighting of value keyword in property event accessors
filter only wm_quit messages in winformsgamewindow
work around dns problem on the build hosts
merge branch testsfix into relpath
rename property to allowgameplayoverlays and update xmldoc accordingly
only allow spaces between the token and trigger character
add accept license to the configuration
only include docs with modules
add try catch block in contextobservergenerator for generated code to support unit tests
terminfodriver be more tolerant to failures and fallback gracefully instead of aborting
update annotations for implementations of ilabelsymbol
add support for foward slash in image source
make scaledcontainer private
merge branch master into import stable scores
some typos fix
allow absolute urls in plugin debug links
update ssdp ssdphandler cs
add licence header
fix google shortener settings
don t suppress unhandled exceptions in debug builds
add doc comment to related methods
fix missing using namespaces
handle null namescope in binding
add tests where t is from the containing type
allow changing capitalization of usernames
add frame ancestors to csp on authorize result for form post
update hosting to account for dataprotection api changes
make objectextensionmanager instance protected set
since arrayrange is now what the tree builder apis expose arraybuilder can be internal
add missed file
change dlna reporting to itemid
check in missing files
changed accelerometer to make use of a static vector variable
fix build errorfix unit tests
modified error message
fix usertableviewcell seperator inset
react to caching api updates
add tolerance to file time comparisons
fix for out of bounds exception
replace br tags to br tags in fulldescription properties
recent list delete shows system confirmation dialog
merge branch master into js better identify
chore adjust for sample async ignore
add integration test
xthor add legacysitelinks
xml comment fixup
fix invalid handling of cc bcc fields in mailtarget reported by bert on the forum
reacting to caching api review changes
return maxlimit in autoquery metadata service
update show in explorer task to use existing helper method remove redundant helper method previously add
dispose mappreviews and map packages
remove hardcoded database name from graceperiodmanagerservice
adds securityholding security get property
allow apply to be called multiple times sequentially
fix appveyor errors
skip it if we reach maxdepth
fix possible null reference in navigationrewritingexpressionvisitor cr andrew
reenable social tests
fix testcaseplaysongselect not working on consecutive executions
move code down
apply same fix to legacy accuracy counter
add loading spinner while editor screen loads
also update the ide handling for computing the default descriptor severity based on all options
reacting to hosting api changes
fix new mouse buttons causing boxing
product tag pages
changed filesize reading to use await when reading the filesize
iptorrents fix link selector
updated the asp net sample to enable detailed hub errors
use iwebhostenvironment instead of ihostingenvironment
regex for headers
subtract a result from count if its been reverted
add tosha256hashbytes and tosha512hashbytes extension methods
add iis variation to nano smoke tests
authentication stateless user lookup func returns a claimsprincipal
fix transitioning of unranked label
assembly version update
shows real owner of disguised units to allies
disable teamcity flow logger for now it s a little too messy and needs refinement
make platform methods private
fix musiccontroller raising trackchanged event twice
initial hack at finding a converter given two types
updated threadagnosticattributetest in threadagnostictests cs to skip enumerated types
add misc fallback
remove redundant comparison
remove bind in player
fix unreachable code issue
fix duplex proxy constructor arg names
allow specifying type names in publishresponseswhitelist
merge branch master of github com mono mono
add obsolete attribute to getstate overload in keyboard class
torrentech add support for aac category mapping
use presentationparameters rendertargetusage to control back buffer clearing or not
pls stop crashing
made reactiveentitysystem subsystem public
call startrequestcontext for mq requests as well
add back retry overlay
add back missing verbs for backwards compat
fix recording cannot start if delay is set
fix layout issue where tab content wasn t taking up the whole screen
marked a couple of tests as notdotnet as they blocked nunit completion on net
fix parameter order
improve flush api docs
fix the elapsed time in game time tick
remove probing for now
fix a typo
merge branch master into feature keyspline
add todo comment
remove unnecessary allocations
fix output of object initializers
add more driver cookies
change test to match new behavior of calling authservice with specified providername
use precision almostbigger
xthor fix and simplify category handling
fix a create bug
merge branch master into spline easing
continue if disables
typo symetrical symmetrical
fix null pointer
defining the tmp vars in a single line again
wip still working on nested routing
say sharex directory indexer
remove redundant spacings
ensure sender of correct type in addclasshandler
source code refactoring
fix testcase dodginess
using statements not inside preprocessor directive
add blocking call back in for eta purposes will remove in a following commit
update code that deserializes xaml numbers
sl core add missing library internal
update xmldoc with accurate information about the model
use extension method
fix mania not getting its own selection handler
generalize error message when creating a server fails
inherit beatmapsetdownloader from component
chore adjust point and size debugger display
use begininvoke to avoid deadlocks
add ability to adjust game clock s speed pgup pgdn a bit hacky probably need to expose rate better
fix recent projects list
add containsepisodenumber helper on episode
merge branch api filter of github com crobibero jellyfin into api filter
rmoved extra whitespaces
fix merge conflicts
forgot that change for opentk game window fixes
fix plugin old version cleanup
popup amend comment
fix an issue with updating advanced settings
fix banana generation on catch specific maps
update geolocation for android
reacting to configuration changes
read build from versioncode
allow empty video encoder profile co authored by cody robibero cody robibe ro
add expectedbutton check to guardordergenerator
specify calling convention for sqlitedeletedelegate
tweak notification message
fix nullrefexc in sqlmembershiptable
remove unnecessary null check
chore adjust page background update test to use display scale should fix test failure on ios
disable false positive date parser test
cache the func we use for filtering
exclude empty item from completion list
fix regex for dotted property names
add rich comment explaining our behavior
fix speling monkey
updated call to new httpfile to include key parameter
make sure the tray menus got translations
make mocktogglecommentexperimentationservice partnotdiscoverable
mono debugger soft updated closed disconnected state when connection is force closed
fix spriteicon loading textures on the update thread
check callback first
feat dragdrop add missing support for dragstarting and dropcompeleted event handlers
pipeline tool fix quit item
add margin to expanded build output cell
remove unnecessary null check
remove errant todo
fix a typo
add back project disambiguation for source items
fix a bug where y axis and unit width labels were not placed correctly
add idictionary isnullorempty ext method
use typeconverter correctly
fix a bug with direct restore
make float2 immutable
reduce spinner tick and bonus score
remove some usage directives
display shortcut key for reset files changes
fix missing override from historicalreturnsalphamodel
set channel media type
torrentleech fix date selector
contentequals already does a referenceequals check
attempt to fix more failing tests in prodcon builds
set usedestinationvalue only when a resolver is explicitly configured
fix spurious this isn t a server order logging on player disconnect
move to background thread
bumped assembly version
fix build break associated with add portable to packagemanagementservice
xbuild disable some tests which fail on wrench
fix path bug in storeorderevents
don t exit when we don t have composer for now
update name reference
cardigann add support for magnet field
fix slider serialization
more changes to datetimeoffset
fix drawable error message made more clear
react to caching api review changes
create empty ssl directory if using own certs
fix addbasetypes method parameter name spelling mistakes
only go into failing state if previously online
correct mutex release
use lambda expression instead
merge branch master of github com duplicati duplicati
fix renderlayer resize
winforms add example to xml doc of createbrowserwindowinfo
httpwebclient add support for referrer
we now get the form icon from the exe resources
handle null groupby s
add nestedcall case to check that innermost node data is being retrieved
all attributes can now be used for classes interfaces and structs
fix a bug with some version of sqlite that did not correctly detect the implicit column names
update button usage
merge branch master of github com servicestack servicestack
tag helpers required symbol option is enabled for select input tag helpers
fix misspelling of extension in intellisense comments
make ruleset non public
better error message
rename test and add workitem attribute
add plugin configuration downloading
schedule updatefilter calls to avoid operations occuring while at a sub screen
change background for scalingcontainer
git fetcher chmod pre push hook after clone
make responsecontenttype setter public
aircraft reservespawnbuilding style fixes
logexceptions when loading ruleset dlls
minor test fix
remove unused var
fix welcome intro test failure due to no wait logic
use same headers for remoteip as asp net request
set default keybindings for jukebox to stable s ones
rename forgetten copy paste codes
add fallback to normal note image
namespace update loginlink in app template
add terse syntax to c making the last semicolon optional
update bloggingapplicationautomapperprofile cs
revert icctagdataentry equality operator this seems to catch more duplicates
fix chat for defeated players
fix windows style path separators not being migrated on unix systems
fix failure to decompile certain assemblies
update usermanagerspecificationtests cs
fidex line area bug
trash a duplicate property
remove debug code
save the replay metadata block when quitting
null check add
updated to correct build error
merge branch master into toplevels
fixes crypto rounding
color squares values were not carried over when copying products or variants
add real8value tostring
add selection changed hook
write html to file for debugging
case sensitive music brainz search
flush response before disposing buffered outputstream
fix a problem in xamlil resource path generation sometimes resource path look like avares assemblyname dll resourcename xaml
clscompliant fixes in consumer
fix phantom mcv again
make textawesome have a shadow
add a lint pass to catch mirey s bogus autotarget wiring cases
fix racecondition in test
remove unused field
remove unnecessary casts
naively send messages that the user types doesn t arrive yet
fix unit config tests by setting config back to default
adjust repeat tail fade in to match stable closer
fix another bug with inserting duplicate metadatasets
cleaned up using statements
dont allow text selection to be lost when context menu is opened i e for copy and paste to work
merge remote tracking branch upstream master into convertiftoswitchtrivia
fix zero length hash models incorrectly creating a unique hash
fix swagger ui document title
source code typo fix
fix threading bug in loadedassembly
fix multipleruntimestest singlebenchmarkcanbeexecutedformultpleruntimes
fix syntax error
disabled by default include untracked files in stash
move the timer positions
don t stop alchemy server if it hasn t already started
add accidentally deleted code
the next alpha version
binding redirect for system text encoding codepages
delegate remove throw when passing delegates of incompatible types
fix events inheriting unique attribute
merge branch master into js fix diffused animations
restore newlines between top level map yaml blocks
demo drawer handles scroll bar better
run child updates for nested hitobjects when parent hitobjects are masked
specs fact attribute inherits from observation attribute
improve behavior for asrawjson so it s embeddable as a js assignment as is
make backspace cycle bases binding more obvious not all of us know the whole character set or care to
fix flaky test
review co authored by jérôme laban jerome laban nventive com
fix possible crash
fix condition in gituiextensions cs
add date filter
set the world owner for color picker logic
fix compile errors
don t load method bodies if no cflow deob is used and no types remove
add missing sortedsetcount methods on iredisclient
add collection id as set by mcm
fix remove nzbs org newznab preset
make sure to clear property grid
treeview dont raise change events on the clear stage of setting a single selection
expand try finally
restore original values in fadecolour method call
fix searching for movie after it is add from a list
microsoft build buildengine update failing test
fix new rss sync threshold
linux setup fix compile error
remove unused variable
sharing the sampler initialization with clear
make field readonly
use full path for analyzer references
move throwing if cancelled after logging that fact
add licence header and remove unnecessary whitespace
fix beatmap processor not setting combo index on first object
provide a cancellation token when joining _allrenamelocationstask
indicate if managed decorations are needed via platform
fix assertion for the bootstrap compiler
add post update notification for ios users
windows setup reverse button order
guard from null
remove unneeded changes
remove unused usings and add end of file newline
fixes _data never getting initialized in soundeffect constructor
fix vertexstride was not set in vertexcontent if model contains only a position vertex channel
add more shortcuts to literepository
preparing consparam and use path combine in favor of string concatenation
invert canshowcursor conditional to fix cursor not showing in visualtests
fix inverse access restriction bug
merge branch disabletest into genequalsfrommembersui
remove usage of snapchat icon
no need for player var
remove unused method
some tests interference i believe
fix parallelism spawning the correct handlers
extract method for marking cheese
windx fix mousestate xbutton1 and xbutton2 were not being set
fix typeofexpression translation
avoid ienumerable allocations when flusing data to sqlite
hooked up new project menu item
remove slight sillyness from ralint output
update descriptionxmlbuilder cs
use icollection string contains to avoid enumerator allocation
don t least at player just yet
fix the contentwriter again works now on spritefonts and song duration
don t add array binarysearch dependency for unreachable code
set featuredependencycontext tenantid before using
don t try to autowire string properties
make tree retrival optional it s not required for non c vb languages
fix nre in sse getsubscriptionsdetails
isn t desync fun
create recording folders on start
update arraybuilder t toimmutableandfree to prefer movetoimmutable
fix build on desktop platforms
remove unneeded nullable check
source code comment changed virtual categories should be virtual directories iis
graciously handle transfer status reporting for syncjobitems
correct spelling in comment and some spacing
fixes bug in updating live custom data sources
log not everything is a time interval
fix inserttext in overwrite mode
fix scheduled tasks not honoring time limit
support in foreach
reacting to kestrel extensions
it seems that getentryassembly might return null
use linejoin round in freehand pen
add if statement to specifically verify that it s a nameresolutionfailure and not just a webexception
remove config from addauthorization
do not propagate positional input to subtree
addin pass navigateto none to ilspy as command line parameter
popuppanel add measure logging
correct wrong fix
fix code format
increment errors if there s an exception
clean up state properly when a carryall s cargo has died
the carryable should not unreserve itself
test commit for madu
remove dead comment
update alphastreamsslippagemodel cs fix bug to return slippage value for hard coded symbols rather than return as a percent
disable stage reset hotkeys for new files
changing name and assert in response to peer feedback
add overload to httpresult
rename unit tests
winmerge diff command fix
linux bubbles clean up
use additive colour adjust constants a bit
replace variable name for integers with i j k etc
fetcher fix freeze on cancel
remove unnecessary option in filters
capture mouse when passing through event this moves the implicit capture from the overlay layer to the target element
remove unnecessary call to string format
fix wasm fix clearing the data of a path
update istorageitem cs remove not implemented
remove unnecessary documentation
return viewcomponent result directly from a controller
dynamicvertexbuffer constructor fix
remove dopostprocessorstep return value
merge branch release into dev
fix kekekeks notes
config never let user name be null or empty
add javascript content type to the response of getall action
remove excess assignment of the g and a pixels
merge branch results screen animate once into results screen applause
remove deprecated properties
fix compilation issue
hotkeys use ctrl c to copy commit hash to clipboard
reduce the scale of background blurs
add git status porcelain method to the gitprocess class
show net runtime version
add missing finalizer
fix missing using statement we now initialize mediaengine with audiostreamcategory gamemedia
merge branch master into fix mod crash
slight code clean up
docsdocumentcontroller fullsearchenabledasync route kebab case
make nodes sealed
corlib update securitycriticalattribute test for updated mobile api
minor refactoring of pluginmanager class add null validation
sampleapp enable error logging in release config
system update flaky test
expression body and spacing
fix taiko drumroll bodies behaving badly with edge alphas
game content is now get set
ms build pass defaulttoolsversion from projectinstance to build request data
fix merge error
fix username display on beatmap panels
respond to code review feedback
fix test completion item description
add task yield to avoid synchronus deadlock by ensuring asynchronous execution
turn on oop by default
fix null reference to fix ci
making query internals visible to wox
add more debug information in to di resolution
only add spans that overlap
remove overrides of getinitialmarginrequiredfororder
changed nameof parameter to correct one
add an test to cover the existing code path before this fix
ogg end of stream fix
update image sizes
config add includeregistryprograms for list
fix invalid redondent cast
update mainstartup cs updated based on feedback best to keep everything consistent
remove redundant cast co authored by peppy pe ppy sh
fix merge conflict in conversion cs
update a r calculation
update pdbwriter cs
use patterns in more places
update key counter font
consider synctimeout when forming default connecttimeout
use camelcase for field names
address pr feedback to avoid leaking pooled hashset
remove default attribute from being generated in add ss ref
ensure that _customunderlines was assigned to before disposing
add removepostfix removeprefix unit tests
update test to use useperrequestservices
update plik cs
don t crash when zipfile getentry returns null
correct error messages for singleasync
exename not set for tortoisemerge
make methods private
begin implementing default app menu for osx
add debug info
fix nre in loadedassembly lookupreferencedassemblyinternal
api without bots result in current bot instance now same like every other command api asf exists for older behaviour
fix openapi file schema
set a default zoom level that isn t the whole track
unskip a test as it has been fix in dnx
add missing doc comment
restore old auto popout notification overlay behaviour
add limiting max keyframe interval when full transcoding
don t show default judgements in front of objects for now
fix merge conflict
dim column backgrounds further
do not compute dimensions or blurhash for remote images
passing in existing syntax facts service
add a test to ensure ioperation and patterns play nice together
replaces exception for error logging when ondata for custom data is not defined
update newcommand cs
fix tests to work with different localized strings
add the map to the exception log
add idletimeout and housekeepinginterval defaults
update scriptlisplinqtests cs
adjust precision for failing test
only mark buffers readonly if they re part of a project that supports enc
update src editorfeatures csharptest invertlogical invertlogicaltests cs
re enable branchgrid after it has finished loading
fix poor handling of cache directories
mono debugger soft fix the ilinterpreter to catch argumentexceptions from objectmirror getvalue
set statuscode before setting statusdescription this will prevent the statuscode setter from overwriting the statusdescription
try providing default config for environments that want to use web components but not webframework
fix build for permissions
remove debug leftovers
converted rootpath field to property on the genericfileresponse type
save up to two value lookups when applying layout constraints
textbox add empty pseudoclass
add imdbid to tv search
fix broken tests
change database to unblock tests
adjusted logfile name
merge branch master of tig migueldeicaza gui cs
make sure the repository hosts menu shows as soon as plugins are loaded
guard against null
merge branch master into dev
fix execute on grab script if release group is unknown
xml duh don t hide the stacktrace
update src features core portable embeddedlanguages dateandtime embeddedcompletioncontext cs
update json cs
android mediaplayer fix nre when source is not mediaplaybacklist
typo in syntaxannotation class description
handle null outputfileextension with null conditional operator
fix cache in datafile on checkpoint
more colour consistency across formbrowse
fixes bug infinite loop in qcalgorithm ondata slice
switch to using networkinterface networkinterfacetype instead of networkinterface loopbackinterfaceindex to detect loopback ip addresses
introduce iabptaghelperservice process to reduce asynchelper runsync usage
changed connectsignupresponse to be an enum
a pointertype s sort order is never usable today
merge with andy s changes
the dispatcher wants the access
implement cultureinfoconverter getculturename cultureinfoconverter cs implement getculturename
expose calculateaccuracy for performance tools
fix race condition in stop after connecting fails
fix problem of first install
fixes default leverage for all cash accounts
make contrail fields readonly
fix openapi json validation error on online swagger io
add istoplevel property
flesh out comment
one everything is fine per project is enough
make resource extensions public
fix os dependent substring
fix viewmodelviewhost to build on mac
maint remove commented code
switch to keydownhandler for faster
fix hardcoded test grain class namespace to match the reality
fix wrong commit
fix xmldoc references
plinq re enable tests that were previously ignore ed
adjust graph sizings to match web
monkeydoc log parameter status when an error occurs in the html generator
initialise reactiveui on afterplatformservicessetup hook
don t load plugins starting with microsoft
add more xmldocs
spritebatch cs comment improvements
fix concurrency issue in base app host
don t synthesize a parameter list out of nowhere
trim quotes from dsm version parts
parse symbol if exists
fix units visually moving to a bogus subcell via moveintoworld
log exceptions in apiconnection tryrequest
add deprecation notices to renderunit and rendersimple
reformatting to fix some weird line ending problems
rename container to match settings term used everywhere
undo dup fix
use lookupsymbolsinternal when we don t need to diagnose failure to find
make type private
excluding tests for silverlight
fix codacy pr quality review define the locale to be used in this string operation make tryformathexadecimal a static method
return correct romanian string for hours ago
add options services to startup
fix only truetype fonts are supported this is not a truetype font
don t attempt to install update on a non installed version for now
handle the case where stringcontent is empty
check if argument node is provided
playto gettransportinfo fix
update customerservicevalidationtests cs
fix icon crash
update fillmodel cs
improve ib connection refused message
fix a spelling issue
remove capturescreen binding does not exist for open powershell
make beatmap edit button reload beatmap without mods
fix race in test database creation
make type sealed
fix toolbar location loading bug
add unit test to verify functionality of commandlineargvw in linux
assign world from the constructor and make the field readonly
merge branch master of github com ppy osu into general fixes conflicts osu game beatmaps beatmap cs
add unitofworkmanager and currentunitofwork to abpservicebase
merge branch master into fix path control point drag
danishbits add legacy site links
remove tests with de ch culture
use list with post sort for nested hitobjects
statusicon change wording
move enablecors to request dto
minor test fix
fix incorrect handling of download failures
foobar trigger connection
fix one more case of incorrect offset
quick docs fix
hard type incoming ruleset
update decrypter detection code
use private instead of protected
fix serach control background never being updated
fix border only margin
add debuggerhidden attribute so vs doesn t break when break on user unhandled exceptions is checked
extend usingtransform checkresourcetype to support non generic system collections idictionaryenumerator
merge with latest master
merge branch master of github com duplicati duplicati
allow null rank
add an explanatory comment
should fix issue when history fails to capture a download item
don t rely on default relative size of scrollcontainer
assign default bitmapfactory
add explanatory comments
normalize rating to fsk
fix null reference to threading backgroundcontext
merge branch release into dev conflicts tests microsoft aspnet signalr functionaltests server connections connectionfacts cs
remove unused using
remove contenttypeserializer attribute from arrayserializer
remove unused bit position constants in readerwriterlockslim
remove unneeded sleep
update iserviceprovider references
also add the wan switch to the public system info
allow beatmapdatabase to be constructed without a host
mac eventlog allow smaller sizes than relative initial size
dispose update scheduler
dispose apphost in integration tests
more fixes to botched merge i got in a hurry
fix null ref
rename startupcompleted to startupwizardcompleted
add operationtype to remaining irequest mocks
expose some internal methods for reuse
ryujit is always avaiable for net core
make sure timestamp is in utc format
system servicemodel security exclude security forwarders from mobile builds
make sure item has a place to store local meta before trying to store there
make incrementinvalidloginattemptcount async
update src workspaces sharedutilitiesandextensions compiler core namingstyles serialization symbolspecification cs
merge branch master into fix speed adjustments
make the mission a bit harder add a heavy tank to each wave
switch deferred test scheduler to eventloopscheduler which is closer to dispatcherscheduler
fix error building audio on macos
merge branch bao
use global counter for gensym
add another test
make musiccontroller a focused overlay
use case insensitive string comparison in msbuildprojectloader properties map
merge branch master into fix multiplayer crash
cardigannindexer add tmdbid as auto optional field
remove unnecessary di
simplify setting of active tab conflicts cefsharp winforms example browserform cs
remove obsolete comment this is referencing a method that has long since been deleted
use skia arcto method
handle x png
disassembler fix output of class type names in security declarations
only search in types defined in mainassembly remove old debug message
fetcher add exclude rule for autosave files
remove enum newline
sealing some more sinks
use initial final rate instead of speedchange
use stringisnullorempty method
closing merge dialog inspite of conflicts happen
fixes bug in securityportfoliomanager getbuyingpower
add scrambled to the releasegroup cleanup list
use a better name for hashset in extracttraitdocscommand friendlytypename
remove unnecessary and incompatible finalizer
add ignoring features option into genproj
reacting to mvc changes
catalog api pic url
check for palette file before start gif encoding
remove unused methods
classify trailing trivia in xml docs
return iservicecollection from addrouting extension methods
lock the dictionary signed off by dotnet bot dotnet bot microsoft com
trim line result to avoid trailling r on linux
white space trimmed
add documentation to pixelaccessor
more formatting tidy
fix smart indent at top of file
corrected a typo in a test method name in renkoconsolidatortests
cleanup the sqlite journal file if it exists can happen after an exception
handle hover blocking in volumeoverlay to ensure correct hide scheduling
update emby server implementations library userviewmanager cs co authored by dkanada dkanada users noreply github com
add region for new errors
add some comments explaining what is what
replaced rngcrytoserviceprovider with randomnumbergenerator create
gecko add more overloads for renderable to support scaling
minor re ordering to make playfield onjudgement be called before scoreprocessor s
remove failed job logging from the worker class it will be implemented as a filter
replace possible decimal separators with invariantculture numberformat numberdecimalseparator
fix null check in jobmode applycore
use the constant
don t tell all players about my yaks
remove unnecessary tolist
remove unused factory method
only search for mef parts in dll files
don t think we need this expire call
update xml documentation
re enable basic taiko beatmap conversion tests
remove unused using directives
fix ci inspections
skip flaky tests
ibuilder moved to the namespace microsoft aspnet builder fix it in the musicstore
taghelper suppress label generation if nolabel specified
fixes misspelled method name
minelayer limited ammo support
stringbuilderpool allow fastpool object to have larger capacity without being discarded
overwrite existing content handler if a new one is add with the same name
add using for datamember filter
make non null
workaround race condition in failing test
fix bad naming in morningstart industry name
remove unused enum
a quick fix to only fire back button events on windows phone if they weren t handled fired yet
update taskscheduler when config is saved with netimportsynccommand
demonoid add redirect error detection
fix typo in comment
fix wrong quality being detected scan will be slower though
update annotations for types implementing ialiassymbol
fix canfocus not being true by default for tableview
bugfix open terminal emulator in mono linux
chore mark richtextblock not implemented add missing attribute this ensures that the generated docs and intellisense warnings are correct
reorder for clarity
fix disposable dispose will never be called
a bit more cleanup
handle non syntax tree locations
fix a dns test
add a note about showmetrodialogasync
fix unit test authorizationinterceptor_tests
set _selfreference to null after remove it from the collection
jsondeserialize int to bool
basic auth iuservalidator returns a claimsprincipal
fix bindable propagation potentially making swells strong
add container exists t in funq to check existence without resolving
fix logic error on middleware
merge remote tracking branch origin features managed notifications into features managed notifications
batch_separator_not_appended_if_batch_header_empty test fix
format score same as expanded panel
winui comp tweak backdrop blur
react to resolution changes on android like immersive mode
remove unused using
whitepsace clean up
they are system endpoints
prevent handling closing click in popup
force to stash untracked files too
chore xaml check for frameworkelement not iframeworkelement in xaml source generation
fix up formatting to comply with mono coding guidelines
importertypedescription properly handles null typename values in equals gethashcode
increase wait to make the test more reliable
formcommit no summary for deleted submodules
remove unused variable
update iconfigurationmanager cs
fix nullability warning
two more tests effected by di exception change
fix serialization bug
converted holdings constructor to use security type for custom rounding
rename internal api for consistency
reencode patch string to losslessencoding
add ie delay
same default value for parameter
fix drag handles not showing on now playing playlist items
save score upon map completion
argumentbuilder accepts ienumerable objectid
add a workaround for ikvm unfixed baked method tokens
be resilient to not having a containing type for a backing field
remove uncommented line didn t mean to push this
facades add cominterfacetype type forwarder
show exception message
make blurhash be computed during regular scans if it was not already computed
give the body a size
make the description label optional
remove return user input calls
jacketttray fix running detection
set quandl authtoken in live data feed
fix test impacted by merge
adjust transitions of drawableroom
account for panel height when remove as off screen
fix typo in log message httcontext user merged httpcontext user merged
fix unfolding for current type
make the broker disposable
recolour ranked status pill
minor issue fix in google analytics plugin some locales were not remove when uninstalling it
repo ignore file ending with a tilde
changed how the hashes are re used so it is hopefully more compatible with windows crypto api
unused using directive
add missing semicolon to fix the build
microsoft build engine update missing method exception type
on exception when calling js from net return js stack trace to net invoker
mark properties as canbenull
remove unused usings
fix failing test
alter app cookie configuration code samples
match usage of in rule messages with graphql js
add omiting not valid groups to getgrouplist
add iapphost name
adds leandata getcommonticktype
update identityroleeto cs
fix spelling of cancelled
fix a typo that stops users from setting the closebuttonmargin property of metrotabitem
add explanation for isflipped
cleanup extra spaces
fix roundedcordner clipping for contentpresenter wasm
fix typo on scoreprocessor xmldoc
marking some classes as partial
renaming missed variable
nit remove page from categoryname
remove catch of exception that is not needed
inline vector2 operator and point tovector2
don t convert byref parameters
add enum support to orchardswitch
remove bogus firstordefault deref in teleport just use trait
add check if user locked screen orientation if he did lock it then screen orientation is not changed anymore
small performance improvement
fixes english error
mdoc update properly map system intptr to native int for ilasm similarly system uintptr maps to native unsigned int
don t attempt to provide a navbar if there is no navbar language service
add monodevelopgtk synchronization context as consumption of roslyn in vs for mac uses this context
refactored grain props slightly fix null serviceprovider problem
update to sb config retrypolicy on topicclient as well
tickle setauthsuspect if authenticateasclient fails
avoid indexoutofrangeexception when fc match returns no matches
use botname identifier for new steamclient logging context
adds a check to disable music controller s seek
access windowmodes via ibindablelist
update src editorfeatures csharptest classification semanticclassifiertests cs co authored by joey robichaud joseph robichaud microsoft com
remove result nullable check
add support for about delegate
fix resupply anim not always finishing properly
made autoorientextensions non generic
use better format for api
call endrequest in asp net servicestackpage on dispose to dispose of ioc deps
remove dead crap from plane
fix beatmapgroup initialization not correctly setting panels to hidden
add comments for staticconfig allowfilestreamuploadasync
remove unused usings across project
correctly use the defercreation attribute
improve error msg on snapshot failure
exceptionlayoutrenderer prevent nullrefexception when expcetion is null
fix method name after merge
respect locked fields
target writeasynclogevents ilist asynclogeventinfo to public
genericurlrouteprovider should not have the lowest possible priority cannot be overriden
ensure selected item is visible
graphicsadapter subsystemid can be negative
fix opening the scriptwrapper lua file as read write
don t touch old log when disabled
don t process images with the wrong pixel format in mapcache
mark platformextensions internal for now
adapt the utility commands to import crates as well
materials were not being assigned to the geometry
fix an issue where some data was not correctly reported after an error was detected
add xmlnsdefinition for perspex controls shapes
settingspage onpageshown takes some time in scriptsettingspage add wait cursor
adjust wave colours slightly
do not call socket disconnect if the socket was not connected
fix code style
windowsbase implement vector formattable tostring
add trivia to the statement
support vertex objects with custom z coordinate
add error dialog for new folder ation
add negative test
fix applytemplate call the onapplytemplate synchronously when applytemplate is used
fix reg fix the keymodifier propagation
always ensure writers are flushed
remove debug stmts
thumb adds pressed pseudo class
move check up
update mediainfoservice cs
make use serializeas content true attribute for ilist properties
reset mouse confine mode on input setting reset hotkey
remove a hello world from the code
add comment for entityframeworkcoremyprojectnamedbschemamigrator
use the mapper in the base class
small fix in jpegcolorconverter while browsing the code spotted that an int was accidentally passed as float
if full scan occurred do not append previous set
fix the multiply blend mode
use invariantculture in mathconverter
update cron cs explicitly define space as char instead of converting it from string
update mediabrowser model updates packageinfo cs co authored by cody robibero cody robibe ro
timer should only be initialized once when using runoneframe
revert shortening of universe symbols
source code refactoring made sendemail method of eamilsender virtual
adjust visual display of the placement curve
do not restart the accepting socket if the silo is stopping
add script extension method exmaple
change the way volumecontrol handles global input
merge branch master into judgement size fix
slight optimization of updating the tree when deleting a node refresh one row rather than the whole tree
updated runbase to swallow exceptions in send
add xmldoc for updatebindtarget
add lookup generic method useful when reloading module
no need to updatesubmodule if failed to create new branch
config use file writealltext to write initial config
merge branch memory routed event dynamicinvoke of https github com marchingcube avalonia into memory routed event dynamicinvoke
even if we re resuming from scratch try to load our saved state first
suppress event never used
merge branch master into newcode
reset the client state when being moved to spectator
fix non windows build
getdisplayname should either be internal or public
fix ci inspections
merge branch master into speedacc rework
back to upper case since it works fine and keeps it consistent
fix also mdlc layout renderer
check culture name
only instantiate stack if needed
handle serializing unresolved analyzer references
githublocalrepositorypath option add to source code download
allow specifying tz as offset in market hours db
use async overloads
commented in the assert true to make the test fail
make setproperties private
remove else and add comments
source directory is the base raw data folder
implement hp increase for taiko
add failing test for clearing templated child s parent
remove todo probably not a consideration going forward
fix diagnostic name for connectionerror looks like a copy paste oops here the diagnostic name was incorrect
add watson information
update tag on cancel requests
remove accidental change
remove topmost from notificationform
fix unhiding the cursor on mac
remove stop event log
remove old commented out code
keep cached linked doc info
don t show a kick button for the host while ready
replace a static nop action that s unique per generic type parameter by stubs nop
parenthesize constant expression
use webassemblyruntime invokejs in hosted mode
fix build error
fix duplicate test name
system fixes a warning
add comment for header extraction
demonoid change to public
forgot to remove something sorry
update failing mobile test
fix comment and remove magic numbers
make combinestats an ext method
fix ios build
updating packagerepository comment based on pr feedback
add overload for memorygroupview t in memorygroupenumerator t
disable optimized comparer for byte ushort enums
remove unused using
remove extra space
fix issue with imdb lists some not all not parsing properly causing an issue when trying to add them
fix for error message
revert truncation in derived class
fix comma delimited array model binder
remove unused code
bcl free strings allocated using marshal securestringtobstr using marshal zerofreebstr
remove unnecessary code
recursively retrieve files
remove unused assignment
fix a problem when one of the parent directories contains a dot
merge branch master into cfd
exclude appbuilder test from running on mono
fix test failing on mono an assertion was failing on mono fix with a different assertion
show nodes for image and text drawing oncreating
forgot to set webrequest contentlength in async when there s a request body present
only pad textbox after collection is created
increase retry count in autobahn tests
property typo fix
add temp winrt hack to fix relative paths
add a blocking check to production canuseexit
improving test sql script loading hack
fix the possibility of a double direction progression bind
event log add some debug rules
intro fix possible freeze by not calling a window manager action in the main thread
updated unit test to check for specific date value
add sendmousewheelevent to wpf control add sendmousewheelevent to iwebbrowser with xml comment
introduce one more constant cleanup
use length instead of any for an array
chore clarify boolean precedence
fix water animation direction
small fix to test infra throw exceltion when element asserter is used without query sorter
sn improve signing error message
fix logic for shouldsavepropertyhistory
increase timeout in sqlservermigrationstest
filestatuslist can now display submodule status icons
addurlsegment overload addurlsegment overload method to allow pass object as value so that calling tostring on value is not needed
dispose config managers ahead of time to avoid database errors
fix incorrect variable usage
add extention method for indicating if the route it executing or the module is being constructed
updated igatewaylistprovider documentation
merge branch master into fix importing some beatmaps
don t replace explicit redirect param if exists
expose changehandler savestate via interface
compact framework doens t support getentryassembly
quick search shows red when no match also add a wait cursor when selecting next match
fix calling applychanges in ctor again and xna deaults to windowed
add descriptions for laserzap color and hitanim traits
add pre query string middleware function so users can run there own query string logic before ocelot does its mappings
fix basicrepository tests when not on utc common
build the ecma pub is one directory above
fix mistyped error code
source code typo fix
add todo entries
report unknown codecs when failing
minor updates in comments so it is clearer
remove unused namespace imports
adds log with python version
move sizing to ctor
better error parsing for redisnativeclient
move first beat offset to beatsyncedcontainer
use pattern switch
check user login if policy is null
update usage of testsink
fixes bug in marketonclose fill model
add new hostcontext virtualfilesystem to hold single read write vfs source
bug fix for the las bug fix
use rotationabsolute to calculate spm
javascript encode the url we generate
maint add unknown type for platformtype
fix harvesters harvesting resources from underground
remove duplicate comment info check in toggle block comment
fix minor bug with passwordauthenticator
fix pause tests
avoid enumerable count on hot path
rename for clarity
display commands in alphabetical order
small grammar fix
do not consider waiting harvesters to be idle
use public api so its source compatible with drop in
twofactorproviders only for two factor response
update comments for handleremoteauthenticateasync method
run navigateto unit tests in oop as well
fix failed decompilation of catch when if the method is async
xbuild workaround for null log service which should never be null
make test scene usable
add additional keywods
mark osumodtestscene as abstract
controller do delete the config file afterwards
use correct size for overlay
update codegenforeachtests cs
merge branch master into fix square graph overhead
function of changing fadetime
fix playercolorpalette in editor
enabled reward points by default for new installations
remove redundant check
remove allocation constrained test for bokeh blur
revert piano reverb to a separate sample
remove unnecessary method
add samples for the expressions that will be changed
normalize ui api configuration of an extension property if possible
torrentday change default url back to torrentday it
correcting response file syntax to require as intended
update src analyzers core analyzers converttypeoftonameof abstractconverttypeoftonameofdiagnosticanalyzer cs co authored by cyrusnajmabadi cyrus najmabadi gmail com
use local port in update checker
merge branch master of https github com quantconnect lean
don t unlimit fps for visual tests
add some colours to the compose mode timeline
fix move series logging a failure and a success message
fix navigationview panetitle may be null
add failing test
disable github comment posting for prs
add jsonproperty on game s properties
merge branch master of https github com jimbobsquarepants imagesharp into progressive playground
make drawing screen transition properly
always spawn the native error dialog
serialize datetime and bool with invariantculture
brasiltracker update description
add error message packet sending to backtesting allowed messages add default case for unspecified messages types
use shared headroom
fix filetypechoices prop to use a private setter
add an overload of raiseandsetifchanged that lets you use private backing fields
better command line parsing
trim target path properly
fix output of local variable names in debugging symbols
update src features core portable completion providers abstractrecommendationservicebasedcompletionprovider cs
increase padding in formbrowse tab control headers
remove check on static accessors preventing getters on base methods
disabling test parallelism
yet more boxes
hack around bad interaction between firing anims and idle anims finishing
merge branch master into fixes scroll to selected item listbox
create wiki page for creating spelling dictionaries
remove redundant check introduced in previous commit
add copyright headers
work around a non interactable window bug exposed by earlier graphics refactoring
disable display for not
siloterminatedevent set should be the last thing in the silo shutdown sequence
fix sample to use new interface
revert a breaking change introduced without any explanation
fix property name
loops restrict return statement pattern to return simpleidentifier
disable warnings when implementing deprecated api members
repo git use commit message from file to avoid command line conflicts
remove line break in musiccontroller
specify logger type
route template in http verb attr
normalizepaths in tests
add conn string name for all modules
fix bogus base cycler
remove unused usings
minor refactoring of source code
putty config file path was incorrectly parsed
fix a typo
missed an extra set of on an object initialiser
fix nre in start game missionbrowser isdisabled query
clean up sample
move a variable out of for loop
move online id null check to only bypass metrics lookup
remove unnessasary backwards loop from cut and pasted code
fix method name
playing a soundeffectinstance from a paused state no longer adds a duplicate entry to the soundeffectinstancepool
use the centerposition as the starting position
add an explicit ctor reference to a ctor used by reflection for monotouch
update the last quickui
replaced mspec tests for head requests
needs a new
fix typo setting custom directory for rtorrent
side fix raisesisopenchangedevent
merge branch master of https github com hangfireio hangfire
update mvctemplatetest cs
add support for simple entities in navigation
pass currenttime to xungrabpointer
attempt to fix slow mem leak in live feed
make previewdialogworkspace internal to facilitate other testing
made viewport titlesafearea read only as per documentation
avoid potential null reference exception
ignore firefox and chromium temp download files
combine construction and assignment
fix case where parsing failed on empty element tags
remove the apply in the catalogfilterpaginationspecification this is so i can commit this to the repo as an example
add exceptionhandling for the ssh agent killing
remove superfluous namespace
fixes bug in qcalgorithm calculateorderquantity
add param allowedextensions description
remove surrender button for single player games
name methods to prevent accidental misusage
remove taskscheduler useless finalizer
deleted unused line from persistentconnection cs
fix getdefaultconstructor for winrt
renormalize tangent frame after transformation
add duplicate security check in addsecurity addsecurity was not checking if the security had already been add
merge branch master of https github com mediabrowser mediabrowser
catch applicationexception in directory watchers
rename currentclaimsprincipaltenantresolvecontributor to currentusertenantresolvecontributor
add template markers
fix selections while scrolling timeline
add titlechanged event handler
ignore failing runtimetest
replace delegate field with event
hide notification when playing osu hide notification overlay when playing osu
fix and add xmldoc
done processing warning
we now throw on unhandled primitive types
remove unnecessary newline
renamed directory backup in the getdirectorieswrite method
adjust transition for smaller sizes
add optimization for timeout infinite
fix tolist return type
remove wrong assembly attribute
add comment explaining the query string building
update vertical alignment of chroma area
fix harvs dragging to refinery even if it died while turning
update analyzerconfig sectionnamematching cs
remove unused using
a couple of small changes to property patterns per code review
globally fix number parsing on non english systems
don t close same type twice when base call closes this declaration
use null as the regular arrays
address previous pr feedback
merge branch master into fix selected psuedo class getting cleared
update tests cs postgresql syntax error insert into tcat breed name values breed name
be resilient to null
disable a corlib test assertion which is no longer true
add link to issues
add validatoroptions propertychainseparator updated propertychain tostring implementation to use validatoroptions propertychainsepartor to allow pluggable support of different join separators
oops forgoet the bin path probe bit
fix diagnose message for cache server if not cache server is present display none
further changes in nopbbcodeeditortaghelper
restored requiredpermissionname variable
move functionaltests lockholder to the unified version number
disable unrootedconnectionsgetremovedfromheartbeat because it s flaky
rearrange tracks in ondrag
add trailsequences to bullet and use that instead of sequences
use right kind from syntaxkinds service
add fix all test
add escape parsing to parseurl
remove unused field
handle httpexception explicitly and include comments for workaround
switch context is now minibutton
merge branch mania beatmap conversion into mania distance object conversion
remove spurious debug line
don t crash when it becomes disabled after activation
add missing overrides and fix a codestyle issue
system date fix defaultvalue expression serialization
react to iapplicationbuilder server change
serverservice check mono version during startup
react to test regression in httpresponsemessage reasonphrase
use https for api requests fix missed urls
make apiaccess s state only privately settable
don t consider the type
increase load delay on panel backgrounds add back missing fade
minor caching issue fix
fix incorrect key texture lookup
mcs private fields of structs can have type of another private not imported type
allow access to hhf internal state
commit for testing build error
iterate over ienumerable before disposing
null safety in the case where there are no hitobjects
add error message
codedocs clientconfiguration fix build warning about api doc comments on clientconfiguration registerstreamprovider method
pass samplerstate to drawbatchgl20 so it can be activated so linearwrap etc work correctly
update file name and update summary
fix focused control display in devtools
restructure code for clarity
add clarifying xmldoc for rotationabsolute
add safety check for when no objects have been assigned
fix some more typos
remove top level environment variables from default config
fix bug in prev
fix failing test occasionally getting wrong ruleset beatmap
change depth of hud yes this is intentional
merge branch macos of github com mattleibow essentials into macos
fix takecover dependence on attackturreted
safehandle mono friendly
fail fail fail
remove ifdefs our tests compile release sometimes
ignore note adds in the log
repo get the remote changes if detected on a first connect
fix up failing test
assign irequest dto in executeservice
update parameter with default value
do not skip editorconfig integration tests
fix pattern matching
reduce background fade opacity
assignvariablenames getnamefrominstruction also handle ldflda directly
fix not remove seeded download if it was manual imported in some cases
react to loadpixeldata change
documented paml hack
remove map method
updated strings to jellyfin server
maint remove test console logging
use absolute path for back color
add api s for remove plugin links
samples don t get paused
windows fix event log scrollbar height
system fixes failing test
prevent hang on connection start
react to weblistener settings api changes
remove some unused osx handling
fix datagrid virtualization in some situations
let s complete the circle by add the canreload method to the public generic most interface also
capture init errors in initerror
adjust for dpi scaling
corlib test assemblyname culturename test that assemblyname culturename matches net behaviour regarding assemblyname with invariant culture info
fix type option in icsharpcode decompiler console
add spacing to diff icons on direct panels
remove unnecessary dispose call
add resizemode enum
repo add todo
improved speed for verifying database consistency
switch from fileattributerecallonopen to fileattributerecallondataaccess
move thread memorybarrier call into task for every case
fixes stopping looped sounds on platforms using xaudio fixes stopping looped sounds on platforms using xaudio
remove comments for obsoleted code
note message var
add notnull annotation
ignore nullabilities when comparing parameter types when generating exceptions
update lastmodified when saving the session
dispose socket in test after synchronous connect
merge branch master into menu mvvm
remove unnecessary new specification
add braces to lock personal preference i want to be sure that everything is wrapped correctly
fix test after changing method header logic
adds missing licences
add restart button for multiplayer replays
set rtbxcommitmessage readonly to true
debounce music controller seeks
remove unnecessary local definition of colour logic from taiko judgement
ensure absolute path used in output dir
remove unused code
moved pipelined to correct place
add minimumsamesitepolicy none in the cookie policy options on the templates
make rootobject and seasonrootobject internal again
use the legacy authenticationexception
remove connect log from test console
set the windows auth scheme from the microsoft kestrel constants
handle uninitialized call to getorders
add system channel type and use it for the channelselectortab
add response code descriptions
handle no hierarchy case when getting the navbar imagelist
corrected the type of the actioncommandparameter property
merge branch livebuild of https github com heejaechang roslyn into livebuild
merge branch master into lowercase back button
fix csharpconvertiftoswitchcoderefactoringprovider rewriting handling of return values
trigger shake if hitcircles are hit too early
change text colour to black
finally no register
update textboxhelper cs executes updatesource on target on cleartextbutton clicked
rely on bindable s formatting rather than setting a default
fix off by one counts
fix last binary page
make spawnmpunits initial unit configurable
file save all
fix error message of single assertion
fix powerbar scaling
skip hanging test
update array_heapallocvsstackalloc cs
use maxvalue instead of positiveinfinity
suppress code analysis message for idisposable in test
netcore reenable some methods on runtimetype
fix comment in checkboxwidget
add configuration to testapplicationenvironment
it doesn t have to be a class
disable logging json migration as it is not ready yet
add missing using statement for logs
set the enum values explicitly
pluginrootintroductionpage remove isinstantsavepage true
add modifiable xmlserializableserializer xmlwritersettings
remove useless container
fix incorrect coordinate mapping on fruit explosion
update dynamicproxyignoretypes cs
use implicit system configuration for encryptionpolicy requireencryption
forgot to add these colors
thread safe list clean
avoid test frame being the precise time of slider end
remove unused member on serverresponse
make slimmenu more opaque
improve error messages for formatting assertions
increase default thresholds
httpwebrequest endgetresponse already does this
clarify comment and log actual search string
fix missing isdisposed property on soundeffect compatibility with xna framework
ignore initial value of autocomplete textbox
refresh revisions after edit settings
remove unnecessary using statement
system fix process tests on watch
fix some variables
fix a test failure
add convenience ctor to the pagesequence class
fix build error
avoid performing eager selection constantly on add ranges of new children
fix imported count incrementing on failures
help the clr find the assemblies it needs
add unit test cases for formattedstringvalueextracter
fix missing cref not letting compile
don t double remove dead actors from the world
add missing prepare phase to eval
use any type assertion on object assign to bypass type checks where not configured
fix list only the files relevant for the current folder if provided
remove commented out method
fix overlooked assignment
fix missing select to full document return
remove final traces or runtime and source sourcediagnosers
don t build engineinstaller
made multiselect properties non browsable by the designer
chore fix handling for single line opening xaml tag when add ignore node
increase spin count on test to not end up in frequent sleeps on multicore machines
fix empty cast conversion from dynamic type object type argument
fix multi header title not aligning correctly when changing screens
add todo comment to remind to revert
