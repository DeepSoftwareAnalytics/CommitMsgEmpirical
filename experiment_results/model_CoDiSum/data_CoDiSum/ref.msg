fix npe when back is pressed multiple times on profilelayout
add htmlbrowser plugin
use its own logger
ignore pce thrown from document commit callback
reorder javadoc at clauses to appease the checkstyle gods
fix hdfsunderfilesystem listrecursive test for hdfs profiles
hg rename root sync control setting to fix ise after class change
remove duplicate method in defaultplan
had a system out println that in the code from a long time ago
fix combiningunilateralsortmerger for reduceexternalitcase
chaineddynamicproperty booleanproperty requires dynamicbooleanpropertythatsupportsnull
don t reset suggestion table on start
don t process offline mode revert events as rename move
android remove deprecated api
fix null check for baseobj equals
fix boolean return value
add new line on sharing text
remove unwanted import from utils java
refine last segment calculation
change nomenclature for no ssh key found
remove an obsolete todo
skipping test which occational fail on ci
add sdk type check
fix process id check don t return true on threads
fix errant search and replace
remove unused code
remove log messages
fix backlog tracer dump traced message to match node or route id
don t increase the number of unread event log notifications on every compilation
remove original request from the json representation of responsedefinition
add javadoc comment
add instance id and current timestamp to the temporary untar directory
mocked local channels acquire buffers in a blocked fashion now
don t backup night mode setting
update slow test for ssplit changes
minor remove useless in enum
add a tostring method to scannersupplierimpl
explicitly initialize static final boolean to fix build
detect labels on blocks which are not sets and report an error
cannot compare binder references for rc display to unregister
add cancel button to sort
handle missing source doc can happen for e g plots when publishing
remove the system out println of the test
fix npe if there is no accelerometer
fix validating to be true by default and to set namespaceaware accordingly
remove answered todo item
ignore a rowseq when it s add out of order to a span
fixes a problem with our port allocation policy reported by christian schanes
ignore failing tests
fixes file headers for checkstyle
tidying up changes
add tests to filenamegenerator
remove pointless exception
remove useless generic threadpool size setting
fix unit test
sqlite fetch id fail
fix test failure
use new context name in testsuite
add better javadoc
add editor to main menu
improved the docs by add more copyspec samples to the copy task
add constructor that also takes a looper
handle division by zero
remember selected show list tab again
only get the contact list once
target the deployment class loader with new proxy classes
correct hit the interval test
improved id generation
remove redundant setvalue call
fix almost invisible npe in terminaltransformer when loading a non existant class
add exception handling around stale frames for model metrics constructor
cleanup towards immutability still extends a class that s not immutable though
fix unsupportedoperationexception from mockcontext
close the base socket when bindlisten fails
add completedtasks to compactionmanager
add test comment
add metrics hashealthchecks
updated asked state for moved functions
hide keystoreparameter builder setrandomizedencryptionrequired
add failed spanish parser state split
remove none option from threading dialog
fetch the simple name of the class instead of doing regex dances
reset mmessagecount before executing select
add unit test for setting referer header
reduce garbage pollution from virtualfile getfiletype
corrected tiling for rotated images
reduce default rmd outline width
fix compile issues
enabled close test in grouptest java
with the newest appcompat oyu have to pass the toolbar to the actionbardrawertoggle
remove unnecessary code
grapher is producing corrupt graphs
remove outdated todo in defaultserver
remove dead variable _hdfsactive
don t bork safari by remove spaces in front of parens in media lines
restore lost access level on eclipseclasspath noexportconfiguration
move getcachekeysourceuri out of cachekeyfactory interface
don t overwrite files if size differs in projectextensionmanager
add throwable cause to table partitionalreadyexistsexception
check vfs sanity action
checked monitors length in moco server creation
icollection extends collection
remove unnecessary semicolons
fix double alert on posteditor mediaerror notification
fix the length of the secret key
add new required config setting to test setup and fix junit deprecation warnings
commented keyboard input
fix import style in moco request hit test
check command type for power status monitor action
add missing super call as spotted by nulano
allow camel elasticsearch to work in an osgi environment
changed path so that it could be accessed through a class loader while running in the osgi framework
notify listeners that the script was changed when changing enabled state
used set content length api rather set header directly
remove debugging statement
android change viewappear event
move back line with progressmanager checkcanceled need refactroing to perform unit tests easily
fix tiny typo in example
remove some javadocs for nonexistant parameters in vector2d methods
fix an eclispe nag
let toolwindow notifications not affect the ide notification icon color they spam too much
send failure response to spout instead of doing nothing for the case that acker receives fail before init
soften up result tolerance
fix requestbodyhelper body inflation
decrease the key progress increment step size to allow finer control of
initialize gdsl extensions on a thread which probably has a read action already
revert javarootpolicy extraction
allow idle tooltip to be nudged right but not left on cursor change
add the missing onstartasync event
add embedded checks when building the ridmap
fix prop resolution for jetty
add warning when the config file cannot be found
fix rare npe in dropboxmanagerservice
incorporated pr comment
fix broken test from rebase
remove company default copyright header doh
remove break lines
fix documentui s perf tests compile failure
enlarge the chat write area
updated forge requirement
fix concurrency error
fix tiny javadoc problems
make datakey testkey to be the source
return the assumption of overrideable getters as pure
reduce color of background fading a bit
add scope for google plus login
future proof getlabel converts all the whitespace separated words in a string into capitalized words
fix bug on using range on queries when index are involved
fix minor typo in fastthreadlocal javadoc
create context service for components that need it
fix two logic error
fix nullpointerexception in ninepatch constructor
make implementation of getparentclassloader consistent with the other implementations across the code base
remove a c p comment after counter loop we never get here but clean
fix eserviceholder receiver delegation method call
add a tostring implementation for filebackedreloadinformant
enabled show history for directories
remove unused constant and add a constant for the rest api prefix
fix logger in librarysourcesconfig
fix test case
fix spacing and remove println
textview not add text to accessibility events
log a message when exec is made to an unregistered plugin
add getoperations method for convenience
remove copyblock from blockstore interface as we are not using it right now
add message setbody charsequence
add an empty argument after bash c arg
sync engine preserve file timestamps
fix the build
fix a bug where requestdisallowintercepttouchevent does not get called
don t assume n in blocktemplate
add the shared metadata
fix an incorrect conversion in timeshiftsetplaybackparams
fix typo in the exception message
renamed leftcurlycheck to abstractleftcurlycheck
clarify requestmappinghandleradapter javadoc
fie alreadydisposed exception
rename reference to property made through an accessor
don t run this test in h2
outline widget gets own copy of scope tree avoid errors resulting from concurrent modifications of scope tree
discard the buffer if the protocol is unknown
make ut testnonaccessiblefile locale independent
change spiderapi to ignore empty context names when handling scan action
fix a deadlock due to wtf in basebundle
improved some comments
skip minified javascripts when calculating duplicates
release the scratch pbuffer surface after use
do not show immersive mode help in touch exploration mode
remove byte array from db for attachment
add indexer isconnectedandhealthy method
add test for gauge processing
improve notifyadaptersubitemschanged fix issue which occurs when deleting sub items
disable encoding not specified inspection in console
add getters for border color and border width
fix checkbuild error due to javadoc reference to hidden class
add webkit namespace to layout inflater
send progress message in dataasynchttpresponsehandler
fix an issue where clearing transient views could anr
delete the best models during deletion of each cross validation model
protect the ui from getting stuck when no tags are associated with a query
put all contained messages into getmessage
remove async todo
improved shutdown procedure to be safer
add a few java docs
changed loglevel for client hz instance creation to warning
rename pluginmodule registerconfigbeans to addconfigbeans
processtail also moves caret
fix compile error on linux
animcontrol clearchannels now notify the animcycledone on every listener for every channels before clearing the channels
add atmosphere license
fix null pointer in valuecomparator
fix hive connector partition lookup
fix arrayindexoutofboundsexception when parsing hive timestamp column
restore layout without scrollers
fix an issue with stop in pipeline
add missing incubating
fix example in cookiedecoder javadoc
uncommented ejb exception mapper test checked the test is passing for me when running against gf locally
some property names may not have description
fix ringbuffer add security interceptor params
remove useless local variable
add targetapi annotation to keyboardutil detach
add missing javadocs in coremetrics
fix renaming from rgb to rgb java and others
remove usage of sourcepathresolves deprecated methods from thrift
fix check style
fix quality flaw
remove unused generic type
fix weld arq bug
changed method to load upos data file
fix npe in entertransitioncoordinator
fix syntax error in code example
add note to docu about connect proxy http request issue
reduced message level to debug since this happens quite frequently now as of the binding re use
add trace logging for how long it took to load field data cache
fix javadoc warning
don t reset charset if not needed
add new line
grid reset fix
improves opengl es support especially for raspberry pi contribution of erkki nokso koivisto
fix javadoc of lifecycleawarechannelhandler
switch the order of doing the init and put inithttp first in launchserver
smallfix add the usage of the delimiter parameter to the getlist method of the tachyonconf class
make the save password selected by default
avoid potential npe
add eviction map consumer test
add coreexternalannotationsmanager editexternalannotations method to avoid compilation error on updating to newer idea
fix bug where break tab icon wasnt getting set
add more debugging information to the awareness decider
add translation for countries and etc in addresssearch
don t allow session cookie fallback to be used with ssl
fix blending on the hotbar
include all ddns when exporting a context
add auto acknowledge factory method
fix compilation error
corrected the header
add more verbose logging
fix nan comparison error
fix testsqlstageexecution testsplitassignment
use audio and video instead of numeric id s for media elements
remove system out
remove incorrect equal to
expose searchorbview enableorbcoloranimation
confirm injections dialog with enter click
remove an obsolete comment
forgot license notice
add test annotations to frameworks common tests
don t treat plugin as third party if jetbrains is one of its vendors
add all the peepholefoldconstants errors to the outofrange group
remove number of tests from checked log line
fix sql command sequence create drop syntax
call startinput on return from sleep mode
remove unused constants in processproperties
fix wifi cell on boot after wifi in use
fix viewgroup caching defaults
fully qualify a javadoc link ref
changed benchmark constant
make app module field private
fix no favorite count issue
fix a bug on concurrent ddl modifications postgres
fix some minor documentation
fix quality flaw
fix indexing of jar roots
implemented getsize to return max ndef size
modified maximum screen size for deskshare to test
fix exception preventing proper shutdown of atmosphere
use new blender version in blendertool
remove bitmap recycling because it caused bugs
try remove ambiguity dur to green as isloaded color
fix javadoc which broke build
fix a typo implying intents must and must not be explicit
handle null bytes from getdata request
fix the ssid display issue
use the proper license on sample code
fix another possible null case
add gwt module definition only when html included
make getstencilbits static
cleaned up warnings in filterchainproxytests
remove embarrassing code
made column name optional
fix issue when doing a dryrun cancellation and spefifying an empty planphasespecifier
add support for drivermanager and jdbc url based pool configuration
reduced the tolerance
add javadoc to abstractjaxbmessageconverter
add facade method to analyzingfacadeforjs for web demo
don t show notifications configurable when there s nothing to configure
autocomplete now checks only table column keywords and ignore all other
fix bug in indexbufferobject
add serialversionuid field with old value
detect permission lost
add a note that spending the genesis transaction should be impossible
fix weighted response mean computation in glm
addressed a number of issues flagged by jenkins
eliminate eclipse warnings
don t use local id in comparison on conflict
fix concurrentmodificationexception caused by trying to edit the world in a different thread from the main one
ignore elastic averaging dl junit for now
fix an npe if available is called before the input buffer is created
continue firefox rdp
don t delete the cache on startup
fix missing metadata when querying the dataresources rest endpoint with a resource name
properly bubble up any exceptions during handshake failure
improve docs to describe close as being last method called
add generic http request factory method
fix npe in discoverinfo identities xmlstringbuilder
fix layout in layout logic
fix sdk int check in textinputlayout
fix default constructor in json
fix non enteralways scrolling
call the callback on sdkinitialize
drop log level
remove debugging print
putall was using default executor service to put in parallel
add planned to status enum
remove unused imports
add notnull annotation
prevent illegal state exception
fix the build error of maintest
fix compilation error with primitive iterator
add null check for calendarwrapper from date
fix a typo
don t edit the generated files at all makes regeneration simpler
add support for doubles in testqueries
fix javadoc formatting
remove badly placed finish call on error dialog
make ontablescreated no longer abstract
fix attributevalueimpl comparaison
fix up checkstyle errors
fix unit tests which included implicitly defined string literal tokens
add tld info caching
using jdk_headers compiler mode instead of builtins it s not necessary in extensionfunctionstest
add pid to anr logcat printout
and handle the null case in the constructor itself
fix previous check in
remove icky import
fix deprecated decoratorbarriers
increase the message pool size
it now makes a default list in slidinglistactivity
remove old incorrect comment
add assertion for inherited visibility in codegen
camel metrics add a route policy to expose route stats as codehale metrics
localchannelregistry should use concurrenthashmap
add case for dds rgba4444
add loadload fence to validate method of the broadcast receiver
set empty roles semantic properly
keep the controller when video ended
add proper locale
update imports for new extern repos
remove leftover debug
all logs should use utc timezone
encrypt null input as empty
add string representations to commentstatus
remove explicit processor interface as it is inherited from producer
tolerate no response message
remove smack non existing method
handle negative indices in array subscript operator
fix typo in inetaddresses javadoc
add seticonimage method
remove references to framework pointericon api
jsp editor highlighting performance
delete last comma to make sure it compiles
correctly call clearcaches method on introspectionhelper
remove finalize override to avoid multiple calls to unregisterreceiver
make crashreporter survive exceptions in pathmanager
enable move directories with packages inside
made partitioncontainer clear public so that it can be overridden on enterprise
updated benchmarkrunner to reference stephenerialization
fix handling in disconnectingstate
add blog hostname as preference summary
add background and border for editable panel
addressing review comments
resets summarizer after everything s finished
don t load the jar snapshot if we re add new dependency
discard double nan differentiation while sorting
add carrier config variable to support editable wfc mode
remove unnecessary fixme comment
suppress illegal argument exception support info
fix drag and drop surfaces on secondary displays
fix variable usage
avoid race test failures
fix null pointer issues in the timerplugin
remove unnecessary argument
fix constructors for searchindexabledata and searchindexableresource
do not suggest templates for ne module
make appleinfoplistparsingtest fail fast if its resources aren t found
prevent intellij from causing checkstyle failures by not wrapping for each
fix new class level watch for leaf or collapsed nodes
fill out method
add convenience method to runtimeenvironment
fix bug in businessaccountmapper
add a method for getting the known states from a model
autoscroll from source shouldn t request focus
fix unit test
changing access to protected
use simpler test api for performing actions
do not enable wal mode for the webview database
fix tests home path detection for teamcity
add getconverters in order to be able to configure project specific conversions
fix a bug where the margins where not updated correctly
simplify the code
handle null package argument
remove unused columnconfig
remove unused attribute
fix naming of jmx stat for thrifthivemetastore
send missing symbols to stderr
renderer is already receiving proper bounding rectangle see javadoc
fix import in sharedlibraryloader
apparently the wiki has wrong data values for the stone button
using regex pattern on copy did not work properly
fix jsonparser getvalueaslong
avoid npe for interceptors using predicate and stop
plugins can intercept urls with
remove bad tangle
test add a test that ensures version current is actually the latest version
also implement new testrule interface
not print ids for postcodes
modified query for contact list to include active chat contacts
fix a cs error of camel xmpp
don t scan attachments when picking the featured image
expose profilerequest clone publicly just marking as cloneable does not actually expose the method
convert anon class to lambda
fix test for unix pathseparator
fix potential java util concurrentmodificationexception using addservice
add danmakufilters class
use directinstantiator instantiate
remove unused local variable
classloading change for fixedcertificates
move comment into javadoc for format consistency
require persistent notifications
catch badly formed urls in webview accessibility injector
add three const strings as templates to generate properties of tiered store at a given level
fix broken helios authentication integration tests
add border title
fix navigation to compiled overridden descriptors
fix null pointer exception
include setter in api
if we remove optional token from suggestion then remove leading space from the next word
fix typo that causes dreaming on battery
print edge addresses for easier debugging
unit tests should use availableportfinder instead of fix port numbers
skip emitting a fix when there are multiple top level type declarations
remove unnecessary singleton instance of managedproxyfactory
remove path from grammar name when present
abort swipe up gesture when long press is detected
add registration for backward compatibility
previous change broke compilation
remove unused imports from webapps
set the default mode to grid view
instead of using a perfect equality test for stopping future column
remove some system out calls
remove debug statement update todos
add global option to camel cdi
make instrumentingclassloaderconfig final
add tostring for easy debugging
fix firing the remap event
fix broken osgi test
create libraries modes and tools folders in the sketchbook if they don t already exist when determining the sketchbook location
remove mock servlet context method incompatible with servlet api version
fix alternate time zone display for clock widget
fix recents to freeform animation
log pool name in warn messages
migrate collectionfactory to containerutil
restore original active protostuff smile entries
reverted the keyword test
add comments to lazily initialized singleton implementation
test httpcompressionit is a real it
handle wrap_content in setviewpaddingbottom
fix bug where was comparing lexer stringval to token and not the token name
fix minor bug in flatten strings
missing empty line between methods
fix spi tests for varchar change
fix npe in javautils
add onreset to geobroker
fix write ops of statistics
add getters for column width and row height
remove unused variable
add break to switch case
debugging the blinking test
don t merge messages over the char limit
add the exclusion to androidexcludedrefs for the appwidgethost thisbash leak
pipes should not connect to zero size inventories
add some checks to prevent json error in console
add log line that could help us in case of errors in the json parsing with queryjson
protect more service dump methods
fix the search header on search results
add max_p2sh_sigops constant
ui application exerciser monkey crashes my app in this place with
evaluate def symbol metadata only once
fix url transformation algorithm
fix unexpected rotation change when re enabling auto rotate
packagelocation should document suppression mechanism
don t consume the event
fix assert usage
tell downloadlogfragment to refresh
fix missing eventlistener import
officially deprecated abstractjpatests
add some more documentation about temporary notification avoidance code using uids
fix tree s equals method
fix typo in permissionquery
decoupling testrunner from idea rt
add cluster name to slop mbean
moved reflectionutil actors common reflection
fix latest merge no thresholds in glmmodel constructor
remove groupby with selector
updated javadoc for exchange
add a hashcode and equals method so that interning annotations that reference items works correctly
remove accidental use of instance in typefactory
don t load document to calculate view provider stamp
convert sample uris to aeron uri scheme from udp scheme
test remove debug leftover
add mediatype for google sketchup
set worldgenerator menu visible options to three
tweak resource path for android
add uncommited methods sorry for breaking the build
change setrotation documentaion
fix upgrade qa
npe fix for syncstorageengine read authority
only clear the caret when in caret selection mode
add logo splash animation in onboardingactivity
fix a couple of places where integer multiplication was cast to long results
minor remove unnecessary unwrapping of variableasfunctionresolvedcall
making user model serializable
sometimes linkedinoauthrequestfilter in api is using redirectquery equal to null
do not resolve call expressions if they are not about regexps
add constructor with default transferminsize to path and file resource managers
fix checks on builder error handling
remove accidental import
fix trivial typo
update uploads view when finish upload is deleted
fix npe when showing multiple snackbars
adds logger prints for the outlook contact query
optimize patternset intersect
add tostring support for null cells to help with debugging
add data binding support for progress
fix helpactivity icons for light theme
ignore the manual test
undoing last change
fix issue where accessibility keyboard mode callbacks were not always called
topicservice should not drop the events coming from recently shutdown members
replaced linkedlist for arraylist
remove raw types
remove unnecessary reflection code from javatemplates
cached string value of connectionid to avoid cost of string format
assert that delayed merge stacks don t contain other delayed merge stacks
reuse the file static asset handler
add support for iscapped in legacy collection
fix conflicting native library error message
return a blank plugin context path if it is null
throw exception if can t create inspection output dir
add war to force cpu on implementations with a broken blur
fix accidental banner regression
avoid warning message at shutdown if globalresourcelifecaclelistener isn t configured
allow null to be cached in memory for unavailable urls
fix two typos in javadoc
suppress unused warning
percolate exception back to caller don t say internal server error
add java util stream basestream to defaults of result of method call ignored inspection
fix bug on collate from studio
rolled back change in jsontypeserializer
filter initialization error will throw multiexception
fix test issue where index is explicitly deleted during cluster state update
add notnull annotations to android xml value wrapper
remove unnecessary space
fix unicode regexp for u
renamed parameter matches to match
cleaning up imports
increased the test timeout
tiny bit more space for buttons
cannot enter data if jetbrains account fails or returns several licenses
fix highlight alpha not being applied in copy constructor
update processing java documentation since output is no longer required
add a public method to set the owner of the field
fix package name of debugcomponentownershipmodule
corrected javadoc link
disallow constant response
removes debug printing
fix javadoc typo
changed druiddatasource discardconnectiion filling logic
fix the broken comparison app due to change in simplecontrollerbuilder
add outline on how cache invalidation could be made slightly more efficient
fix a crash while generating display lists
fix file name validation
optimize the map traverse
remove fixme is already fix fine
add test for non string role
increased amount of time to wait to execute scheduled uaa task
only memory barrier pending publications at end of batch
immediately reset current book hashes value on changing book
fix infinite loop when uninstalling apps
ensure callback called after mathjax load delay
responsecontent to protected so something that extends can see it
use the max response size provided by the httppagebufferclient
save search query as a client property
add comment for eager creation of batch invalidation object
fix jetjavalibcompletiontest on non windows machines
fix missing done assignment
fix speed test case
idea don t recognize additional criteria supplied in invokation time
don t swallow cause if store stats can t be build
documented public interfaces exposed in this package
fix shadowpreferenceactivity to set preference screen
update defaultstringset javadoc
remove an unused import
cache admin client in admin connection verifier to eliminate threads and connections
remove system out
fix element nesting in javadoc comments
further clarified javadoc
fix typo in comment on gziphttpoutput
fix isseniorclustermember when it s used in the middle of the cluster startup
temporarily ignore unstable tests
calculate return type for method domainobject getpersistentvalue propertyname
lazybsonobject pipe now returns number of piped bytes which is just the value of getbsonsize
hotfix for classloading force new loader in all cases
don t print sampling notice if not sampling
add glm1_rho as the default temporarily
add a tostring to treeshapedstack
implement apply new settings to project tree after save setting
handle null from toprunningnondelayedactivitylocked
make processcursor public to aid contentprovider results
modified constructor to be visible only to the package no longer public
advicewith can now manipulate routes
fix unit test on really slow boxes
fix javadoc warning todo is an unknown tag
fix map item annotations
parse episode time using the new timetools helper
add world getname and world getid
add disallow_create_windows to device owner only user restrictions
include deps of prebuiltnativelibrarybuildrule in apk
modify comments to be more accurate
fix entity rotation
removes an unnecessary debug log statement
fix flaky test
missing one xaxis entry
don t automatically choose a path
add final to ep_name
use key to remove static field from mongodb
remove graphstatistics option not really useful
always use goog exportproperty instead of goog exportsymbol for exports defined
remove unnecessary method
remove unused field from phonewindow decorview
skip trip searches when there are clearly no useful trips
initialize file type lazily
modified the second route to demonstrate the problem
specify default service name for coordinator
content value edit fix
add test for findbykey with delete
fix potential npe in objectidvalidator
fix last modified caching for logo servlet
remove debugging code
corrected help button work
don t draw the resized bitmap if the source and destination are the same
fix the issue paddings are not taken into account along the cross axis
fix action mode bug with select show on map in my tracks
remove unneeded documentation
fix testwavseektoend testcase
add test for null values in attribute
do not compare strings with
add info logging if zk config is not specified
fix serialization of getblocksmessage
normalize all the path cache keys
return the top most generated jtype from schemamapper generate
fix some minor display bugs with the friend accept button
update ctrl p location size after controllers modifications
do not process ignored files
improved methods comments
delete previous patch copy
will not use new code in zlandroidimagemanager
change blender texture locator to always set the texture key on loaded textures explicitly
remove comitted system out in motiontrack
turn off override_with_best_model for cv models
finally track down and fix the node exists problem with not deleting noop requests
read access throwable exception for coverage
fix failing test
the user can disable default interceptors by setting true to disable_atmosphereinterceptor setting
handle null uris in content changed events
fix basepackage not add route and interceptor
add debug statement
fix a typo on tvcontract
do not hardcode class name
add override annotations
fix json serialization for rnn specific vertices in cg
add weight updates to the convolutional backpropagation
add rpc for notifying server of following user
provide a hook to check the origin
remove unused code
actually change the place where updateselection is called
update documentation of tvinputinfo getparentid
add method to som to allow for the retieval of class values for each of the neurons
remove extraneous system out calls
add override annotation
adds some logging when there is a problem connecting connection failed
do not update recent projects on closing in headless mode
add carrier config variable to hide merge call button
circle add x y setters
surround content id with and for better gmail support
only tell the reloadinformant about starting and stopping the app if we are in dev mode
remove unused import in parseable java
improve error for bugcheckers without a bugpattern annotation
import asm opcodes statically vs inheriting them
remove unnecessary synchronization
add support for collection types to the verifier
remove icon for hide empty middle packages
outdated comment fix add short circuit eval for
don t add actions without icons to the file history toolbar
add queuedthreadpool getqueuesize
fix up tests
disabling snapshot due to slow response time on abook
log line set to trace
fix potential race condition in test
prevent user from setting tokens in configuration
en adjust test for the renamed category
disable prepreroll to try and avoid security issues
reverted the logging nonsense in hazelcasttestsupport when i run a test in intellij i want to see logging
remove dead local vars
rename variable name for exception from t to e
noticed issue in subprocessexpansion when model of sub process contains pool
remove debug print
make addhook just return when add an exsited hook object
add some logging for relinking head
ninja fix propagate usage
should not remove cont from unions
add spacing in log messages for readability
improved websocket testserver example to include ping pong
provider geturl now can return null so check it and return fallback not null url in this case
fix the initial values of hdmi hal options
add the missing license header s in the camel jaxb component
don t parse iml files
fix wrong booleans which shouldn t change anything in practice but the code is wrong
use log level debug to tell that a tick tuple was received
these tests expect the legacycookieprocessor
temporarily turning off data matrix in multiformatreader until it s done
eliminate eclipse warnings
revert remove unused applicationcontext fetch
fix view count parsing for large numbers
log exceptions on remote search properly dispatch messagelist changes
fix typo in javadoc
keep screen on when playing video
using getsignature for the comment
allow console output to be constructed into a gallery for satellite chunks
fix kinematicragdollcontrol for bullet
allow dual band support
remove mymapdb from double parser
add detection for pa and carbon
add history todos
remove extra space in learnlibsvmclassifieraggregation
prevent some mysterious out of memory error
incorporate review feedback
add a function to hide the close button
pysetliteralexpression extends pysequenceexpression
fix auto import completion from test scope
fix failing tests
clarify springbootwebsecurityconfiguration scope
the green command should now only affect dirt on the top layer
generate default values for annotation parameters in light classes mode
commented out impersonation test
marked some slow clientregressiontests as nightly
handle charsets that are quoted as if they were not trim them
fix bug when a modloader mod is run on a dedicated server and has a
fix minor grammatical errors
include cause in reader exceptions
fix exec tests once more
add a random boolean with chance
add assumption that loopback supports multicast before testing code that requires it
remove two annoying println s
add an isstarted method to client
do not recreate directory
fix deployment overlays xml bug
never show an undismissable new email notification for a null account like used to happen during setup
components can create users
set location extra after instantiating if null
add change to return if user enters bad number
replace infinite loop with foreach
fix null_deviance for xval ed models in glm
open wildcards accessing array elements for write type migration tests
fix problem with missing stateupdated notifications thanks dmitry
fix tests on graphdb
use new functionality of the new iconicsdrawable
fix minor issue in websockethttprequesthandler
add newline between error code groups
add final modifier
stored superclass elements first
add the generic headers to the out filter in httpheaderstrategy
do not merge add check to make orientation calculations more robust
delay creation of dbdecoder in dbport until after the socket is successfully opened
add a note why we keep all documentation in javadocs
add getequalstoken for future use
add javadoc statement regarding enumerability of methods of injected java
remove reference to remove field in org sqlite sqlitecursor
fix some typos
fix spelling error in onviewdetachedfromwindow javadoc
remove unecessary try catch
add check if directory exist and try create it
solved character image related bug
remove unused imports
add swagger module on ws agent
fix a merge conflict
properly deprecate the field
add comments to clarify requestlogger and deadhoststate
disposing daemonlisteners after project loading cancellation cdr
use the correct variable name mviewscount
inset the non overlay appcompat contextual action bar without a status guard
remove content root icon from jetgradle tool window
fix npe in coveragepublisher for views
remove tooltip text
remove redundant call
corrected generated tests after rebase
fix npe in forwardinglistener
add test for multimap interface
fix import in moco global response test
extending the actor test framework to inject exceptions in remote test actors
default update notifications
allow killapplicationwithappid on secondary users
fix typo in mixineventplayerinteractblock
make byte put public in autobuffer
hide wallpaper when wallpaper target gets hidden
fix backwards assertion
remove obsolete cast in sample
add config chosen
ensure short method name used
make socialautoconfigureradapter properties public
fix definition of fall out again
add diagnostic info
delete temp action files
fix npe in gradientdrawable constructor
fire view remove event after a view is remove
revert turned on no longer flaky concurrency interrupt test
add fix for corrupted settings
scene application doesn t gobble up npes any more
fix thumbnail aspect ratio in flickrphotogrid
add hint to user about stop command
skipping en fbstr merging
remove unused code path
fix issue with dimensionmanager shouldloadspawn
shift right clicking with wrench on robots now first removes wearables
elaborate on bitmap reconfigure expectations
some naming fixes
fix wallettemplate to compile again after biginteger coin refactoring
fix itemframes not having a model
bind wallpaper at later boot phase
fix one other problem
reverting final usage
converted onscrolllistener to interface
fix logic that establishes whether the init from request
cancel indicator on explicit call to future cancel
prevent a stack overflow when generating an error message if parsing fails and provide some useful information in error message
fix configelement getpath bug when including root
don t remove exported class properties
fix the last gump failure created this one
get indent options directly from java style settings
print out snapshot values for better analyzing why cacheprojectintegrationtest caches compiled build script fails intermittently
simplifies var names
quote path in case of spaces
remove useless comment and make sure all public constructors delegate to the package protected constructor
remove asserts which fail on grails elements
android sdk suppress unchecked warning in sharedpreferencestokencache
fix a thread blocking problem
d oh placed the join in the wrong spot
you can t change a configuration which is not in unresolved state
fix bad null check
make web socket timeout errors less verbose
add some context to the exception message
close the channel if an exception accours during the handshake
use buffered streams
fix command execution through simplecommandservice
fix cast to paraemeter type in evaluation
tweak immutable diagnostics
landscape extracted mode floating toolbar interference
implement size method of the iinstructiongraph interface
do not instrument equals hashcode for dateformat field
fix sftp upload for change to common io version
fix offset check
clean up remove unnecessary todo comment
do not merge fix back button state issue
fix invalid return value for exif rotation
remove unused import in bufferutils
forget child view state when there is a change
makes most fields of electioncontext final
moved default latency to latency class
fix last compile errors in highlightphase
fix npe when binding is being deactivated
make getters and setters public on httprequestresponseholder
made extension point available
use correct archetype
bleeding add block getlightfromsky and getlightfromblocks
reduce visibility of makelistenable to package private for use in makechecked only
don t draw vertical guide over a collapsed region
launch bugtraq links in a new tab
fix cs issues
don t include class name into the location string for a function
multipartinputstream test initial part boundary incorrectly allows empty string
changed getcurrentframenumber to getkeyframeindex
add new error message relative to static rule and classrule
choke the log statement not needed
fix for not using permasql translation when reading filter list counts
fix merge formatting
check default settings for property
do not allow tostring operation on id
fix targetdirectory to cope with parameterized tests
don t rely on folderclass implementation details outside of folderclass
add checkreturnvalue to predicates
add function usehmacpageprotection to disable enable hmac protection for further calls to open create of sqlitedatabase
fix a keyguard charging status
differentiate ids for remote interpreters based on vagrant with multiple vms configuration
remove useless imports
remove unused constant
handle no matching endpoint for hint target
hide objectgraph s constructor
add license header to manymappingsbenchmark java
include final in introduce parameter preview
fix a buggy log message and make the comment clearer
remove invalid todo
fix little bug with double deleted consumers
finished missing tests
use correct password in test
add only one space before unique keyword
add doc to getid
add support to translate annotation class
missing test annotation
fix unambiguous import
created multi issues and marked function with number
add timeout to pooledbytebufallocatortest
modified the changecursor logic to handler null cursors before closing them
fix space escaping
avoid npe during rdf regression classify
properly refresh file system on getvirtualfilebyname
bootstrap sessionfactory implements infrastructureproxy interface for resource key unwrapping
remove debug message
fix no warning in log if temp file was not deleted
disable thread dump action when session is inactive
remove rule to always retain receiver type as those are now kept in the instrumented type
add id param to uithread
fix initialization order
don t re create a string from a numeric timestamp just return the string we get
fix namespace sandbox executable name
add missing indent
makes a noisy error to warning
remove now redundant comment
fix wrong comments
add missing override annotation
fix notification of disconnected networks
add anchor to proxy url
ignore associated domains when comparing entitlements
add javadocs for settingsmodule
make flags public just so people can hack it if needed
correctly initialize loosematching on read
create griffon library inside write action
make sure accounts have at least one attempt to login after the reconnect
add print statement to in defaultexechandletest to log the javadoc executable used
fix the oom exception
by ven correctly refresh root directory that had been remove from the disk
set forceframework correctly for included layouts
fix itemtouchhelper log messages when dragging or swiping is disabled
reused pynames dot_py
print the version
revert revert avoid extra getobjectmetadata request
downgrading javadoc reference to class not visible from jetty io it causes javadoc errors failures
remove unused map thanks for the tip to jerrinot
add a todo in resourcetexturesource
fix clusterwidecounter asmap
improve whitespace normalization in baksmali s analysistest
remove defunct flag
cleaned up some deprecated warnings
selection scope does not make much sense for analysis action
add stupididty exception for common issue with getminecraftlogger
log detected wm anyway
delete obsolete api in errorproneanalyzer
add some additional output for test failures
don t trim bodies
inlined file resource with file container interface
remove unused import
add some comment mostly to test build
add t getentry long sequence to the barrier interface
remove invalid comment
simplify failuredetector probability calculation
add some debug info
fix sasl anonymous for updated api
renames and moves the cusax account property from cusax xmppaccountid to
remove old unused code emitting calls to non existent methods
stop any running dhcp client before bringup up an ethernet interface
add support for additional android standard view constructors
forgot that cancerous check style
fix clear services api
use single n between description and cherry picked from since git itself does so
fix javadoc markup
delete redundant annotation
improve javadoc for saved username password pairs
perform dependency analysis and mark found dependencies even if there were errors during the first phase of make
fix return codes
add the failure exception itself not its cause when running in process tests
gtasks sync on save should push a re fetched model
check showmode for updatestatus
skip binary files earlier no need to get psi for that
simplifies var names
just make things public since this is just an example
remove main method
parse entities within comments properly
make management port default to server port
align message with regex for abstractmvcendpoint path
fix watches tests
remove unused test
fix null pointer that can probably occur as result of race condition
remove setting of layout resource as it s done in the xml
remove constant only used by tests
add a matcher for non null literals
fix incorrect order of arguments
fix benchmark tests
remove sub blocks
add missing javadoc comment
fix the initialization order in genericscheduledexecutorservice
remove auto reconnect from ui for now
making progressdialog visible on winxp machines
combobox now processes esc outside dialogs too
replace deprecated method call
turn off theater mode upon power button press
fix scroll jump in scrollview
simplify restparamfortextview method
replace logging output from packet tostring to packet toxml
also remove force option from logger trace
allow networkrequest release from the system process
log all interfaces we try to bind before binding
add comment about user responsibility on frame replace
send room invitation if the room is members only and the user was not a member
do not deliver ring event if fg call is live
fix bug nullpointerexception on branching pacts
remove currencylocalemap since it is no longer necessary
exposing io collection api
blank maxcolumns param caused splitcolumncommand to throw an exception
synchronize dhns eventmanager
fix last bug time for jmx integration
fix demo app to always create the job manager
return cloned usermodel on cookie authentication
add a within bounding box search to querybuilder
use compositestoppable to stop compiler daemons
updated schema version
fix up deprecation messaging in bytestreams and files
make lightweightmessageformatter withoutsource public
add fail fast check for support action bar init
renamed runtimeexceptionfailureitcase to taskfailureitcase
add goroot to gdb command line
use flyway s classloader to detect osgi framework
add an explanatory comment to com intellij ide macro editormacro getcolumnnumber
add a comment to explain handling of messages sent by the server immediately after the websocket connection is established
add a comment about why the indexing trick is needed during featurizing for the conll benchmark
remove redundant documentation
sourceformatter check for correct line break
fix test failure
fix a crash when dismissing task edit popup from rotation
fix a wrong variable when logging
remove unused imports
remove window no title settings not used for abs
remove unused couldnotadjustdownwards
add reminder to zap code i love retiring old code
fix eclipse warning
fix bug of disappeared rounded angles after first display
add missing lower bound to action parameter
update unit test to set netty type to be default
testbeelinehistory throws npe in shutdownhook prasanth jayachandran reviewed by siddharth seth
comment out apparently unused code
switch abstracttestqueries test to use varchar instead of date
remove reference to ldapdataaccessexception since it isn t actually mentioned except in javadoc
add constructors from super class
remove poi index support
allow editing of
fix autocraftingtable not respecting the doremove flag
enable optimized readers by default
fix power provider
fix inaccurate comment
temporarily ignore failing new test
do more cleanup on error
just close the socket
add vector getmidpoint
revert add support for checkcallingpermission to shadowcontext
tweaked datapoint javadoc to depluralise references to datapoints where
fix regression in renderscript detection
add more explanation for notifychannelretuned
move gettemplates usage
yole always use system l f in startup dialogs
android update debug
add better note
fix indexing files without top level members by mistake
throw blockstoreexception if the chain head doesn t seem to be in the store
set prev field when reading json
always create toomanyconnections
is not a branching instruction
remove test code
fix null pointer bug as reported by lovexiaov
add pshape3d setvertex int i float p
add test case for getinstance context
allow jsoup connect to parse application xml and application xhtml xml responses
fix text color for hyphenated words
remove the test code
initialize motaspmode to otasp_unknown
relax the requirement that the socket has to be in waiting requests
fix glm accessibilitynodeinfo row column count
fix duplicate registration of execution vertex listener
fix null pointer bug in remote fetch
make getpatternrules synchronized needed due to its lazy init
fix capitalization error in an inner class
fix npe in addimagetransformmetadataconsumer
fix ns in ns resolution in lookupvar
fix performance of datanode hashcode
remove stale todo
check location permission appop in getscanresults
customselectionactionmodecallback can prevent selection mode from starting
housekeeping remove some duplication
remove hard coded time limit which also interferes with arriveby
improved exception to avoid npe in exception message
fix previous check in
fix nullpointerexception during gui initialisation
prevent annoying warning when no max thread gets entered
avoid null mobile interfaces
corrected mistitled dialog
fix the test error of fileconsumeridempotenttest in camel spring
fix a crazy bug with cfboolean
add more detail to the install successful message for the benefit of people who start eclipse with custom vm parameters
ignore sartestcase as it now fails under a security manager
remove static bwc delete index timeout so that the slow delete
possibly fix conflicts with other plugins since dynamic command registration was add
don t throw index missing when updating number of replicas on a closed index
remove redundant conditional
fix merge break
modify the main method
make packetcollector fields final where possible
toolingapi for idea add a bunch of todos
don t return an unmod collection
fix linux man page handling on sy help
remove unnecessary clearproperty
add package comment in cssnode for consistency
remove call to invalidate in onscroll callback
remove unnecessary unboxing followed by reboxing
skip editorscontexttest in headless environment
add lollipop helper to androidutils
fix some packagemanagertests
android revert wxcomstatement updatefinish
fix error some people were having with blockworldvectors
tostring should never return null
fix javadoc as refresh is always asynchronous
remove outdated nocommit and tweak assert to output bad exception message on failure
fixes a problem renaming a rostergroup
revisit paymentbuseventhandler handling
fix appcompatdelegate javadoc
remove compat mode dialog
fix unmarshalling of rid when the value is not a rid
fix maven tests
remove invalid import
normalize path fragment before saving it in package identifier
fix lost text in notifications for failed uploads
remove redundant test case
add no arguments constructor that doesn t register a metaclass
fix cs error
revert add a comment explaining the improved predicate
restrict scope of npe check for bouncy castle bug
add cancellation of createframe
add comiccollections copyright
removes log statement in netoworksender that spammed the log
fix merge error
correctly detect moved changes
fix command line error message
only merge if not null
smallfix made field a local variable in blockmetatest
update sum loop split sum and zeroing
remove unused constructor and adjust methods visibility in delegatingactionlistener
fix bug in tap
fix a crash on resolveinfos null object
use existing logic of getting xmlattributedescriptor by querying xmlattribute
add javadoc to zincrbyparams
remove the some calls to analysisenvironment getderivedartifact and add a big fat warning that that
remove unused code
fix inconsistent synchronization in httptunnelserver serverthread
fix processing of regex patterns in large terms facet requests
fix method parseresponse that should never return null
add option to skip chunk
fix name typo
command line should show all options
fix typo convertor converter
restore anydpi support working around the lack of folderconfig support
add check that repository was configured in annotator
do not automatically enable walk mode
make map legend zoomable
declares rajawaliscene internal collections as final to protect the safety of synchronization blocks and enforce the intention
test use length norm as the tie breaker in blendedtermquerytest
fix text alignment
tidied up cut n pasted javadoc
remove redundant comments
add javax rmi api to deployment
trailing whitespace police
add parameter requirement
remove logic to retry on tx commit on suspectexception
rewrite getfontsize method to support hidpi
fix timeout and javadocs
fix comment typo
improve failure message from testurls to help us figure out why it s failing on travis e g
remove obsolete assertion
remove leftover dashclock reference in apis
do not allow remote peer to arbitrarily size the hpack decoder dynamic table
remove a reference to the android os bundle in instancestate processor
lint suppress some constantconditions warnings
use a logger instead of system out println
add missing directory to clearfilesandcache
tag the eventpoller with comment to indicate its experimental status
add decorator to serverclasses by default for cdi use when not using cdi mod
fix comment in gridviewexample
fix ajp broken by recent refactoring
ensure bufutil release is called even if writebytes fails in writeandrelease
add timeout if the current printer or its capabilities disappear
fix bug in register for filenames w no extension
modified gobacktoactivity to call goback
updated javadocs of player setplayerlistname to reflect that the char limit is based on length without colours
ignore internal exceptions when deleting possibly invalid requests
add regex flag to replaceall function
fix broken test
fix placeholder on signup page
remove system out println used for debugging
remove finished todo
fix sip registrar transport selection
add bannedstops in clone method of routingrequest
changed icon for osmo in main menu
made the keyfactory option public
use the mapplication field in lieu of k9 app
only report valid references
simplify check that certificates chain is self signed
avoid concurrent mod exception remove from the iterator instead
allow user to choose whether linked articles show up in maximized window or in pop up
fix the adapter index to getitemid to include the mfirstposition offset
don t hold a reference to released buffer
fix sysout with trailing messages after a newline
adapted compiler hints in cogroup and match integration tests
pig throws a useful error when the destination cf doesn t exist
fix material constructor
cancel notification on clear
postponed cancelling of orient s timer on shutdown
disable calls to mobiledatastatetracker isprovisioning as there is no more
deferred icons take ii
set fast thumb for downloaded animations
changed defaultcluster getdescription to fail early if not in connecting state
move code not directly related to the update of the local cache outside
add support for discovering side the block is oriented to
fix typo in documentation for defaultgroovymethods xor
set default update interval for fresh installations
enabled one test
fix off by one with json parameters
automatically determine install dir
fix cast to string in fullbackupagent
temporarily undo onmeasure resize video
remove unused code
improve logging make method public
add javadoc for insufficientcapacityexception
fix wrong mock
remove bad comment
fix button size in new project study panel
add debugging for npe in groupedlist toset
add mms to common phone number types
reset min max compaction threshold when creating sizetieredcompaction strategy
suppressed unnecessary warning message
add support for homegear writeonly datapoints
print timestamp instead of age in location tostring
fix direction to point name
do not launch jython with u option
fix enforcemeteredapnpolicy when data saver is on
updating package name
set default screen orientation value to system
fix line separators in jslibrarysourceconfig
fix a prob b w positive negative inf parsing
skip boring casts
fix initial default
improve junit report output
removes a typo from newtkeyinput
reduced log level from error to info when the mmap tries to fix temporary issues
allow apps to hide the soft keyboard even in a transient state
corrected package name in javadoc
normalized type definition tests
add an additional test
reorganize token sets in uncommentedmain
nti remove text that discourages use of nti
minor changed version to beta add shutdown msg to serr
add small extension to parserunner interface
using the js idea of whitespace while trimming a user selection instead of the java idea of whitespace
remove some duplicated code in documentsui
increase orientation settle time
fix errorprone errors
caching weight wrappers should propagate the bulkscorer
rename non boolean method starting with question word
simplify javadoc of runnotifier firetestfinished
add intcast from char byte short
optimize rowtype gettypeparameters
remove unnecessary duplicate cancellation
don t pass a protected op into handlers
do not log the size of uncompressed ce report
adjust integration test to now correct url returned from arquillianresource injection
fix extendedproperties entityiterator
remove debugging code
updated javadoc to reflect proper value of getprincipal
suppressed the usage of the temporary selector in sun jdk to work around a weird race condition in windows
better handle queries when building the session url
add trakt sync activity fetch log message
hide orange map selection when multi menu disappears
test remove one too many suppresswarnings
print available balance in wallet tostring rather than the redundant number of satoshis
optimize text drawing in collapsingtexthelper
clear the log status when clearing the log
add space before cherry picked message remove unnecessary stringbuilder
fix quality flaw pending update
remove unnecessary write
remove public modifier from interface methods
add support to fakeview for setclickable and setfocusable
start fullscreen directly bug
fix nullpointerexception when trying to send message with to text body attachment only for example
fix javadoc nag
allow customizing speed search comparator in member chooser
makes the history menu more native
add missing license
deleted trailing whitespace
add task duration in response for completed tasks
add conditional trace to avoid tostring of complex statements
add missing log line
fix a dead lock where memoryawarethreadpoolexecutor does not release its semaphore completely
bugfix unlocking zip files mid parse
remove log messages
changed debugging statement in bytebufferedchannelmanager
fix a bug where time picker would lose am pm setting
fix typo and remove unused redundant hashcode implementation from gitlab connector
fix kafkastreams smoketest
fix playback of hls media playlists
add padding to the minidrawer on top if we have a translucent statusbar
use embedded packrat for now
add missing feature declaration in youtrack repository
make test discovery errors fatal
make putjsonenum public
fix the gump failures
update the doc string in request server for rapids endpoints
turn off kv speed test
prevent another indexoutofbounds exception
fix bug in cluster psubscribe call
gracefully handle types that aren t supported by jdbc driver
tiny housekeeping got rid of some duplication
updated javadoc of clickonwebelement
fix the javadoc for heartbeat of fsmasterclient
add null check
remove irrelevant persistence property
fix ignored test case
add empty implementation for addtestprogresslistener
fix a warning
add debug info
fix bug where not always wrapping the connection properly
add ptr class to objcobject
remove sharing in registerconstant
remove dead local variable
adds some space between the search field and the photo in the contact list on windows and linux
remove synchronisation from procedureregistry
fix resource leak warnings reported by eclipse
fix findbug errors
fix issues with noticing that a vpn interface is gone
fix the npe when checking mrebootreason
rename disinjection uninjection
updated javadoc on vertex
fix javadoc in testexecutionlistener regarding ordered
add types to base and simple targets
fix crash in viewpropertyanimator for unparented view
make mute unmute work
no match found exception
adds more aggressive cleanup of new chunks
add getcurrenttype to shaperenderer
if shouts are displayed embedded display the episode show title in the
throw error if the tree is totally malformed
do now use the glasspane if called from a modal dialog
eliminate an unnecessary initializer
add vt over ims global setting
fix broken tests
make bottom border thinner
fix misspelling of positives
add shutdown to globalregistry and actorregistry
handle case where there is no timetable snapshot source
add listener before child is add else listener won t get called on first load
add todo note
remove animation in gotohour
fix context class loader leak
have psurfacefx pay attention to the setvisible argument
set classification to the same value as the checkpoint
fix black keyguard
changing info link to wiki page
fix an issue when merging two interface types
ignore jython s garbage in stdout
fix a classcastexception when creating widgets
prevent base neural network weights from exploding on backprop
fix bug on order by with full text indexes
fix typo in activity class reference
camel marshal unmarshal should catch throwable in case dataformat causes an java lang error
eliminate some todos
add executeinorder helper to utils
remove default scale for geo fields
simplified texteditorbasedstructureviewmodel api
add actions for downloading pngs from signavio
replace mongoproperties condition for spring cloud
correctly set the window callback for toolbaractionbar
add break statement so we don t fall through to add too much to boarding time
do not display seasons with no episodes
treat access private like access destination reported by pj houser
put errorreporter checkreportsonapplicationstart back in the public api
fix race condition in compression test
fix compile problems in host controller module
revert previous modifications
fix framework detection in new project wizard
add test for roundtripping hashcode tostring
add storeload barrier before reading the gating sequences
exclude milestone releases from cross version testing
fix a bug where redacted notifications had the wrong height
fix bug in removefailednode
postpone initaccountservice until roster loaded
fix typecast exception
moved error checking of built ins to more direct place
html viewer doesn t need focus
use the right buffer allocator
save all documents add before update and merge actions from branchpopup
remove local lines
moving constants to top
fix potential nullpointerexception in repositoryfunction
remove an unused method
defer building of error message
make newline platform independent
fix translation cropping
add zero padding of mcc qualifiers
generate shortest escape sequence
add set point scale processing to setmessage
add some comments to the enums and methods in the notificationssettings class
add ide error message for return_type_mismatch_on_override
make sure to propagate privileged status when upgrading an app
fix migration complete message
add a class load assertion from the api module
don t steal focus
add test for setcomplete iscomplete
remove some unnecessary code and fix an eclipse warning
fix argument for notnull parameter directory of com jetbrains python psi pyutil
fix handling of preferences for data panel and antialias
add a overridable method onclose in httpclienttransportoverhttp2
fix tiny initialization issue with filecopierunittest
call postinit when starting an extension
fix memberimpl hazelcastinstance injection
add support for and filename args
add int to shadowbundle
use a small font size by default
performance avoid calling virtualfile getpresentableurl in normal control flow
fix checkstyle error
don t capture stderr if we re logging to slf4j
make use of ex constraint
now checking status status_rolled_back on aftercompletion
add a comment that states when comment_id_unknown is used
accept multiple comma separated ids in idea load plugins id
reduce initial number of positions for topnrownumber
improve usage message for adl
don t highlight implicit anonymous class parameters when they are inside constructor call
show warning when failed to detect out node in non trivial switch
add extra test
incorporated spencer s comment about shuffling oob rows
add an empty constructor to the sipcommtextbutton in order to allow to set the text after creating the button
fix crash when deleting all messages in a thread
deprecated yuv version
remove imageio from pixmapblendingtest
add organizer_can_respond to calendars column
make getdafaultlaf method public
improve documentation for configuration transitions
loading litres catalog after an authorization failure
remove unused import
fix sub type recognization for temperature and humidity messages
tried different way to disable test
updated released version
include scroll indicators in overflow
remove outdated doc link
fix compilation failure in downstream testing
git push fix swapped diff panels in update during push info tab
fix a problem with multiple login and still trying to reconnect
implement missing jumptocurrentstate in layerdrawable
add options to the method enum
enable commit action in commit changes dialog in case when only unversioned files are selected
fix magic bug
remove bogus print
moved ratio to top level
maintain support for fieldvalue of type omulticollectioniterator via iterable interface
kill activation reporting while we figure out how it should work
fix initial privildges for views
replaced import of org hamcrest text stringcontains with org hamcrest matchers
make null friendly
add java doc
findbugs null check on logging merge change in patchset receive pack
always use the max score across the shards in suggest response
application crash and phone restart when app
accessibilitynodeinfo makes incorrect check before calling into the system
fix broken grails debug
revert groovyoverrideimplementutil hack
consider first player name in leave command
allow us to set the contact uri without setting the fast track s uri
fix typo in debug log message
add method generators
rearrange raw table related messages in exceptionmessage
passing project to the dialog
fix http method when urlconnection setdooutput true is called should be post
remove unused imports
add setviewvisibility to remoteviews
add javadoc content for checkstyle
add cell output
remove unused field updater from subjectsubscriptionmanager
fix bunches of cs errors
don t show inline chunk toolbar for sweave or rhtml
remove extra line
fix isinstanceofclass documentation
make valuewithmetadata normal public for use in alternate graph implementations
always update quick settings header translation in onlayout
do not log processcancelledexception
fix handling of senddata responses
add new onsearchrequested overload
fix faulty dependency definition
fix compilation problem if plugin is registered via metadata than exported true for transitive dependencies
add sanitization code for the current position
remove char version of isc0control and made non public
remove unused import statements
try performance improvement
fix preferenceactivity runtimeexception with preferencescreen
could not add contacts for protocol providers that don t support presence fix
use flattened configurations from ivypublication to generate ivy module descriptor
fix construction of twodimtable in confusionmatrix
mentioned aliases in javadocs
add existence for saving
remove unneeded attribute name
fix tutorial test
remove unused imports
make logging of exceptions optional
temporarily initialize default column precision scale and length in o h metamodel relational size
add a link to channelconfig in channel
release entire confighashset upon readonly
compile error fix
add send count to integration metrics
fix bug where inputs are updated from action cache but not tracked in skyframe
use linear filtering for reflection refraction textures
add rpubs buttons back into export menu
drop unneeded arguments
catch execution not only runtime
fix javadoc for subscriberexceptionevent
supress verbose warning of illegal cookie names
add debugging info to asyncexecutorqueueoverflowtest to find out why qa is hanging the data seems to be correct
remove some old debugging leftovers
allow managed realmobjects in non managed lists
dispose file editors in tests
changing test port that conflicts on some systems
prevented isbn parsing from happening twice
fix hpc array size heuristics bug on very large heaps
add timeouts for connections and reads
remove once executed
inherit doc for complete builder
add javadocs on closing responses
uses the same history icon in contact list right button menu and chat file menu
javadoc the directory behavior
remove the action center on zero
don t delete bundled plugins when installing an updated plugin with the same id from the plugin repo
update doc comments
fix test wait for host to come up for manual deploys
add paramatized types to stop raw reference
fix potential nre
add waitatbeginningfactor to hashcode and equals methods in routingrequest
customize selects customize instead of viewing
add license header
make the electron beam animation shorter
do not quick evaluate lambda expressions
fix calls initiated with sip registrarless account
do not merge check apps idle states on time changes
remove unused code
add createconcurrentsoftmap method
add test for invalid binding to classloader
fix up a derp in the timing command
remove unneeded line
add isautogenerated flag for axis
make a method static
fix redundant cast in testtsdbquery
made abstractcontrol implement cloneable
do not expose test connection details
clearing ro status on generate action
don t paint text fields rect in case of graphite
fix merge errors to make jenkins happy
add getsipuri for identification
fix unclean backport in internalloggerfactory
don t display thread count in thread view
shorten toolbar labels when source window gets very narrow
fix exception for codecmus
commiting test that should fail but runs green
duplicate statement in basic3dscenetest java
remove unsued conversation vm cache
mark deprecated old callbacks with deprecated so the ide shows it
fix code style
handle the case of a missing child contacts node
add missing argument check for verify
notify the download cancelation for updating the screen and remove the yellow arrow that continues being shown after the cancel
don t include list owner when parsing members on tag save
remove some unnecessary code
cleaned up exception message in text resource
clarify the use of sessiontransacted
mining wells no longer leave behine mining pipes
add workaround for race conditions in testresource
also return helpful error message in http response in case of exception
remove forgotten test lines
make constructor helper method private
used seperator constant in rest ids
re used parametrized direction
hopefully fix a rare crash in the task list fragment
fix copy n paste errors in nativewebrequest
remove duplicate log
add helper for urlconnection setusecaches
add todos and some more complete class javadocs
not to rewrite resolution_scope for references
ignore ejbclientclusterconfigurationtestcase as it often fails
applicationutil add notnull
remove unnecessary final modifiers from interface method parameter
add javax jms
add additional counting test case
add nullable to metadata get
set loaded show title and overview
remove temporay files on ejb container shutdown
improve generic signature of ioactions createwritefileaction
log the full path to the generated gradleapi jar at info
java remove unnecessary check for term status on roll of connection
add method to create sound from filedescriptor
fix accidental fork logic change
add main class to camel core
add removal with prefix to http cache
provided psi file for decompiled kotlin classes
convert a runtimeexception to a coerceexception in the pathtypecoercer
fix case when cache is switched off and record is remove without index update
improved logged message on mmap error
remove todos no issue here
fix keyguard drawn timeout
fix flaw in unit test for gatedeploymentdescriptor class
deprecate unused code
exclude metric names from dimension list
fix format string in camera
fix broken test
dispose hint on inner editor released
allow checkstyle s unused imports test to be used with the api classes
add nan handling for specific hit score as well
honor the islast setting
fix a bug where clicking the more results suggestion with the trackball
fix head vs head commit resolution
fix statuses for contacts in awaiting authorization group
reduced log level to debug
don t save empty script constraints
set editing to false after code cell run
remove t getcfs id
bean validation upgraded javadoc
changed mongodblayer pool to cached thread pool
fix build breakage
prevent warning when sourcing active document that doesn t end with a newline
eliminate eclipse warnings
update remove node message handler
moved a todo from a public interface to the implementation
remove obsolete code
whoops remove this
prevent serialization after being destroyed
platform restores binary compatibility
provide example actions in accessibilityevent setaction docs
add help id
changed files deleteifexists to files delete in fsblobcontainer
fix import in accountutilstest
camel catalog add api to validate endpoint uri
simplify conditional expression
perform automatic formatting
preventing a npe
remove bug on display of log messages
add stronger scare text to the javadoc for autovalueextension
fix configtest failure
clarified assertion which should be fix on idea side
fix broken manufacturer advertising by checking for null serviceuuid
clearing set of usages on reset
smallfix fix documentation of the allocateblockwithview method in the allocator interface
fix file header and remove unused imports
add some more convenience methods to streettraversalpermission
add activator logging
emit correct udc source from desktop package
bulletproof the box class against accidental world corruption and bugs
make notification content clickable for logout remove guest
do not sanitize override config for fullscreen
fix bug where ssl socket would never invoke
retrieve resources for apps even if not installed for current user
fix npe when trying to get an image from an imageid that is missing in the resources
commented code deletion
remove load method it can now be done via querying instead
make checkstyle happy
fix from code review
fix test to test compression support after login has been made
made method take object instead of object and then cast it
make test passed on server
do not override accept
add support to encrypt assertions
baseelementatcaretintentionaction extends baseelementatcaretintentionaction
bug in format of session kill statement
extended the maximum pulsing duration when dozing
fix bug in utils copyof
trim column names on import
ensure imageview setimageresource always resolves drawable
add getserviceconfig to services config
remove fix todo
fix long line
fix nullpointerexception in universalenglishgrammaticalstructure
suppress output for protobuf data if bluetooth is disabled
adds right click menu button to change nickname for chatrooms
improved javadocs for automatic doc generation
log use commit time instead of author time
add additional information to the range check exception
don t mark the session as dirty when trying to remove a nonexistent entry
fix layout problem in console
fix exception in antpathmatcher for leading
fix bad alpine cmd in tests
fix checkstyle problem
remove xposed logging
fix the build removes unused icon not existing on the repository which was causing the problem
fix a regression in httpmessagedecoder due to a mistake during yak shaving
using temporary on timestamp for jpa entity
fix false positive in javadocstyle check on inheritdoc tag
initialize the component before add a new route to the component
fix npe path inside cell tile
invalidate cache according to changing context during refactoring quick fixes
handle unexpected errors when updating indices
temporarily disabled the test until the missing resource is checked in
always return text filetype in annotate selected previous revision
clarify plugin documentation for requiresunpack
refuse update of a system app if it changes its shared uid
adds that audio stops when a world gets disposed
simplified operation signature
fix npe in inverted index creation
adds ignored test case for negative indexhit size
reverted last commit
remove unused import from test code
remove unnecessary code
changed two text occurances of maven android plugin
add some logging for cancellation events
add analytics to track when the app settings is opened
add diagnostic for the assertion
fix hopefully issues with undoing auto format again
fix broken link
add human error message
improve javadoc for tachyonexception unwrap
raw strings need help
scrollbars fix on node expand collapse
allow the dot file generation to be easily disabled
using myevent in testpostmultipletimes
get test working on windows
change the message when the language is changed on linux
fix management center url update problem
remove unused method
get the executor of the context earlier if possible
remove outdated todo
fix droppers not dropping
add missing documentation
change exception log level to error
replace highlightseverity info usages with weak warnings deprecate api
remove unused serializefeature
missing comment header
fix broken log message
fix typing in splitframe return values
make inspection suppressable
andorid revert switch
wanted a return type from the methodinvoker
add documentation for possible androidkeystore exceptions to authenticate call
ignore null actions
add call to system gc to release windows lock on library files
don t start editing on windows key press
add a comment
fix test on slower boxes such as windows
add missing method to testwebsettings
adds tostring to crouton
when we try to add a conact that s already in the list throw the corresponding exception
refined fix for values on group widget labels
remove obsolete todo from charmatcher
fix binary compatibility of resourcemodel setcopyresourceid integer
fix bug where idents were only tracked if inside classes and interfaces fix to include enums and annotations
add ensureyellow call
always use replicas in testclusterinfoserviceinformationclearonerror
remove stats checking to look for app standby mode
add tests for a bug found whilst reviewing the elparser
add nullpointer check in parsebpmnedge
bloomfilter now sets the background color to blacknoalpha instead of black for glow
encapsulate secondaryindex estimateresultrows
fix bug in testvertexttlwithcompositeindex
add logging for unknown state
only actually log the dns events we have
updated loglevel on baseindex
fix test condition check
when looking at several account preferences the last account presence details were shown
update some docs in ultimateviewadapter
include alignment in formatting model dump
add a bogus method in otherwise empty test to prevent build breakage
allow dynamic max height for scrollable toolbar popup menu
if last status stored in the configuration was offline don t register the protocol provider
different id s since filehistories of files of different types should never have the same id
changed the xml plugin descriptor reader to read the plugin xml directly rather than awesome a relative path
fix unit test
add safe cast to scalableicon in deferrediconimpl
add system gc
do not restore template settings instead of configuration
remove the block size from constructor
use a slightly more sensible startup sequence for boundedoverheadblockstore to avoid log spam when creating a fresh chain
make applicationmaster fields final where appropriate
avoid invalid regions
fix mismatched tag problem debug info
fix sending with ctrl enter
fix grails tests
fix too many children on scroll up
fix javadoc formatting
remove notnull annotation
ignore client aborts when flushing the buffer
fix buffer leak
finish reverting change to servletupgraderequest from websocket ssl branch that was missed
use the intended based on the changes to the policy file name for the new system property
organised some imports
increased daemon start timeout for ci server
reordered outputed xml
reset distance after track saved to sd card
do not add a dependency descriptor mediator to the moduledescriptor
add isempty check on sactivenotificationsmap as close as possible to accessing the key to prevent crash
showing all settings on skip
remove a unused import
minor missing options
don t apply outbound encodings
remove wrong comment
delete old methods from the streetvertexindexservice interface
don t save content if ws is stopped
granted creation of a new client object with a new socket when the network connection is lost
enable right margin option
override api longrunningoperation addtestprogresslistener testprogresslistener listener to return concrete builder type
fix work apps showing as not installed for owner
avoid excessive string building when running stripcode
add logging of raw messages to serial handler
fix the build
log stack trace on failure to send message to client
fix disabled resume for suspend all context
use the class shortname here
simplify tostring to avoid oom
prevent iterator allocation in hpack encoding
fix spelling issue
just fix a typo in a comment
remove dead code
add handler for download event
disable strictmode when starting activity
add new method which will collapse all items
prevent concurency issue when requestscoped and applicationscoped are used
remove unused imports
exclude crlf when logging sent commands
add encryptionmode option to camel nagios
allow threads to die earlier
move serializableconverter out of public api
add call to errordescription dismiss after showing an alert dialog
updated code with proper documentation
add since for key_archive
set enum class for preferenceimpl config
move calendar group init code back to main thread
remove automatic back browse error codes from server do not distinguish between excess of permissions remove share or remove file
add rejoining flag to collab params
make filterinputstream abstract
fix a typo in an exception message
fix repositioned issue
allow scripts to generate breaks
add missing fail
fix search shards count method when targeting concrete and routing aliased indices
skip non physical peter
set the intent action instead of overwriting it
use linkify web_urls linkify email_addresses instead of linkify all
make close idempotent in syncingfilesystem
prevent injection of unannotated dynamic settings
replaced sout with serr in some test to reduce noise
csv add includeid parameter
remove unused executorservice in test
use correct set of permissions when updating
fix typo causing horizontal layouts to not layout properly
prevent infinite loop in the phone ui
gettimestamp on psi file doesn t load the document
require basetopactivity subclasses to implement event sending
don t silently ignore multiple contention handlers for the same file lock
allow integration sample to download dsl milestone
make status bar expanding more tightly coupled to finger
remove some duplicated code that isn t required
add execution exception for non specified module
cross schema foreign keys read fix
remove cache when blocks are remove
suppress modelspecificdistancecalculator error
implemented file deleteonexit
add a todo comment
remove logger getter from deprecationlogger
fix typo that treated fullscreen as transparent
add cause message to log when recovery fails
add connecting status
fix a typo testequansignorecase testequalsignorecase
add debugname to membertype
adjust plugin sorting
register new component
implement clearcache for view resolver
fix integration branch in propose instructions
eliminate eclipse warnings
remove explicit declaration of servletcontextlistener and annotate with weblistener
put back old fix for ro store servlet to pick up config correctly
remove unused constructor in arrayschemafactory
fix internalattribute equals
fix npe reported by werner schmierdorfer
remove key and secret from sina weibo example
updated viewsiteactivity and webviewactivity to use support toolbar w o drawer
add test annotation for outerjoin test
add flag to method s javadoc
keep break buttons in sync when changing mode
fix a unit test
add license header
fix caching of extension classloaders
fix client test
add comment for abstract method
remove some unnecessary code
add referesh at the end of rebase process
fix route settings
improve whitespace matching for roxygen execution
fix appbarlayout enteralwayscollapsed with inset window
add max argument to the pump call to only pump length bytes
only hide the filter view when showing the wordpress com sign in button
invokelater s in abstracttreeui caused running tasks on disposed tree
fix dashboardwidget creation in bundlereceipe
explicitly request a data refresh after configuring a widget
fix guardedby annotations with invalid lock expressions
fix autoboxing warning
fix bug in management graph iterator implementation
reduce the number of executors now that we support async write
don t display stars for route points
vcs log when branches panel is invisible it has zero size
add more logging information
we want this in log in release as well
apt plugin do not scan jdk classes
bosh endpoints redirect all the requests with the trailing slash
fix imports to use non shaded version of guava
increase idallocationtest wait time
reduce the number of force calls on storages
fix legacylogstest and make it use correct canonical paths for root files
add listeners to output layer
fix quality flaw missign license header
when closing task stack remove reference to display content
don t flush each time if we use a buffered output stream
fix formatting of pool size violations
fix intermittent unit test failure for
don t inappropriately kill anring drop recipients
add timezone in node system information
smallfix remove explicit type argument in filesystemworker
don t draw a circle via a path
add missing test for my sanity
local rsv ordering fix
it causes leak
fix pivot for webview exit transition
changed exception that is thrown when a class cannot be loaded
improve performance of absurdly large range s
add chunk getworld
fix bug where sometimes multiple global status buttons show
fix issue on closing project
add unit test for decoderesource with options in shadowbitmapfactory
ensure chronology used in parse
add a warning when server development mode is enabled
fix javadoc error
add log in case of error on setting pojo field
fix canvas finalize for the case where the constructor throws an exception
fix a description of a viewpageraction for tests
test that longer strings sort behind shorter strings
improve command managemetn for editrmdformatoptions
add jsonp test
fix spelling in an error message in exopackage
fix touchablenativefeedback state propagating to children
add support for signavio enterprise ids
add update message
layouts fix divideritemdecoration s constructor argument name
remove swing imports from cross platform code
fix incorrect size issue with inline image text
channelservice no longer connects disconnects the channel so it doesn t need asynchronous start stop
updated version mea culpa
avoid dereferencing potentially uninitialized field
don t pool bitmaps with null configs
fix server crash
remove unused code
fix multiple registration for redeclaration errors
check permission for flag_will_be_foreground
fix a bug in peer error logging found by findbugs
reset connectors when recalculating
quick filters fix numbers format
fix nullpointerexception in getview of contextualundoadapter
clitest should extend cleanuphelper
don t start process if no modules
fix cameramanager open exception handling
slide transition jump before transition
remove unused import
add test case that highlights need to switch to harmony decoder
fix gl version getter for es
fix a broken availability of show usages action
add more logging to debug the empty keyguard
revert also overload execute without arguments to prevent compiler warnings when parametert is a generic type
reset the timeout to the original value
deprecate property editor
remove intent to go to other activity on start
vcs log use toolwindow instead of toolwindowimpl
accept only accessible members
updated error message
do not ignore test sources on off for global scope
always show recent messages contacts even when offline and using presence filter
fix the camel example axis unit test error on osx
add assertopeneventually to testmapevictandlistener
remove hardcore for default fg
remove obsolete prunedexcache
add missing since annotation
corrected null check to be for progressbar
changed the log message so that it indicates the directory where a
fix hide quit issue
add test for mapreduceoutput getoutputcollection
remove unused variable
file is not writable exception fix
hide escalate visibility for inplace introduce constant the same as private
inherit preferred size from the actual panel
fix test failure
resource folders handle fix
fix broken condition
don t store test data in field
set recipient jid
allow empty url
fix the build
don t init wifichannels until after supplicant up
make instance final for both classes
fix one small error
rename capacity to length in context of term buffer
ignored test for large requests to batch operation api
add getonlyconstructor api
set layoutparams in reacttextview to fix crash
fix bug with extra warning for css containg el in jsp
every static resource returned a no cache header
maybe fix index error reported by lukasz witkowski
remove all usages of com intellij util ui tree
add missing di bindings in standalone mode
update viewport even if fbo id already set
lagartodombuilder double decodes attribute values
add method to return confusion matrix variable
reverted change for formatter exception fix invalid characters in html a deeper problem
reduce warnings for implementations of iterable when being analyzed
fix restoration notification intent
instanced objects are now considered a single object when generating rendering statistics
renamed a parameter in bukkitutil
fix npe if hystrix reset called when it s already shutdown
pass correct extras to the custom print options activity
fix db transaction bug saving attachments
changed method curly brace placement
remove unused imports
do not load content of arbitrary files
getannotationattributes generally supports meta annotation lookup
add a comment on what this class does
fix quality flaws
smallfix remove unused import in indexedsettest
add type check to multi level command converter
don t add is not null filter for partitioning column ashutosh chauhan via harish butani
add debugging to c
allow three digit phone numbers
fix compile error
set initial attributes on root directories
add check and throw for failed intrinsic init
revert remove tostring method
fix bug in rewriting applicationid with old value
fix recalculate issue
add cell clearactor
fix surround test
user registration should not required authentication
handle concurrent unsubscription in drain avoid npe
make directory getall public
fix expected object type
wip fix path
fix some javadoc typos
hide current soft input when a background user gains window focus
resolve checkstyle violation final variable
take into account the visible title height when scrolling
add extra comments
skip blank lines instead of throwing an exception
replace existing files when unpacking
add missing null checks for convenience methods
remove serializable true from arraytable
add gservices flag so that market can turn on performance logging for specific groups
rename some files and unblock the comment
remove useless check in nativemetadata
replace unchecked cast with checked one and remove now unnecessary suppresswarnings
fix simplehandlerresulthandler by using publishers completable
revert jdbcdatasourcestat changes
increase the timeout
remove miswordpresscomuser from accountmodel and set hasaccesstoken friendly only package visible
skip txt files in box folder
fix compiler crash when doing instanceof on a getelem
allow end of line comments to take format args
typing on line split border replaces wrap sign with line feed assertionerror at softwrapmodelimpl beforedocumentchangeatcaret
fix two minor errors under default eclipse java syntax compliance level
fix bug reported by joe comuzzi
give access to the httpcontext
fix jumpy scroll on initial typing in console
disabling still broken test
add a restassuredmockmvc standalonesetup defaultmockmvcbuilder method in order to allow more flexible customization of the spring context
remove superfluous semicolon
add missing extensions of model interface
fix compiler error in eclipse
remove copyright accidentally included when copying over from my personal project
fix indent style
fix line length
have consistent scrollbar style on the folderlist and get rid of the strange whitespace on the right
do not add introduce constant fix for batch mode
fix integrationtest context caching
move variables into loop in roster waituntilloaded
remove p from javadoc
fix backwards compatibility issues
make public for upsource
and of course use the upper limit to set the buffer size
add support for custom source regions in imageimpl
fix cyclic reference leaks in referencequeue
fix nullpointerexception during sy restore if file history is invalid
fix bug on aggregated functions
omit sun security
set the thread pool size only when the configuration is changed
add manualrangemarker isvalid
remove empty tooltip that shows in conversation panel
fix osprocessutil getprocessid
remove full screen mode of gallery activity
do not adjust child windows placement for fit to display
fix gwtapplicationconfiguration javadocs
add datetime formatted
fix exception text in selectstatement
disable pyclassicstyleclassinspection by default
add web setting for enabling disabling synthetic links
rename a generated field
remove mcallbacklooper from wallpaperservice
fix typo in app_uri
apt compiler should sort eip properties in order that feels more natural
prevent semi initialized cls stub from being visible from another thread may lead to wrong isvalid false
fix bug on super class
remove useless equals
remove redundant disconnect call for cacheword
data transfer wizard
turned xmpp debug off
throw cause of draw exception
allow in inspection description
remove call to deprecated method
fix utf decoding in datainputstream emu
ignore test to get the rc built
dispose previous root model when changing to a new one without events
made signature for annotations to reflect the annotation property
display currently configured wake up interval
commit documents in edt
fix consolation error
add epsilon indarray as a member of multilayernetwork
max walk distance zero can cause infinite loops and is wrong anyway reported by nirmal mewada
reintroduced correct error message for invalid session
just use title label for yaml header
any use of criterion fromrawselection should get extra parentheses when compiled for sql validation
add toast when no account in selector
sdklocationtext uses correct directory structure based on os name
fix crash on bug report
email preference input type changed to email address
followed the intention
only dispatch context menu selection event to fragments which are not in viewpagers or are currently selected in viewpagers
remove incorrect equals method from comparator
releaselock removes lockhandler from instances to prevent memory leaks
show the default publishing account at the top
corrected error in javadoc
bump version for rb
fix selections when text doesn t fit
fix tachyonfile java
deterministickeychain builder add seedcreationtimesecs
minor change the declaration access for a method
change atomicinteger to atomiclong
speed up a bit
fix broken javadoc
use the correct default for scan interval
do not handle selectall
don t allocate a new handle every time
allow exception method to decide if it is thrown
hive on spark over hbase should accept parameters starting with zookeeper znode nemon lou via jimmy
disable css emmet fuzzy search by default
fix appender not locking when getting log position
use more restrictive access modifiers on some methods
add beta annotation to new subset submap methods
fix leading zeroes issue in hms picker
treat connectionabortedexception as transportexception
fix ant build break
add new option on recipientlist annotation
remove debugging output
fix invalidation of transformed opaque views
remove some redundant code
remove a sysout
get key store
fix refactoring copy paste error
fix typo in view id documentation
replace tolist aka the fake friend to buffer
revert accumulo don t run accumulo tests on windows build
move writeobject from vertex to genericvertex
copy the solid color when mutating a gradient drawable
don t force glsurfaceview to render over native views
fix downloads search
don t expose frominternal
now buddy has method isawaitingauthorization
filling fields in similar constructors
fix windowsforwordinposition ignoring windowsize
remove messagereference from local list of messagereferences when
add copymetadatafrom method to encodedimage
tweaked a comment
add default constructor to make usage from spock more convenient
add fallback to default material name also when using ogremeshkey
fix asyncwritter yet again
fix legacysstabletest ks name
add a swap function to the expression parser
add comment explaining bytes reference edge case
don t retain strings in the debuglog by default
fix test for findandmodify
fix missing clear identity in ams apptaskimpl movetofront
reset playback volume when going to sleep
add the number of nio loops created by vert x
adds logging on networkinstance shutdown
add fallback for integer
simplify generated code for boxing void
improve option description
do not merge don t crash when canceling alarms if a direct listener is present
remove unused emptydelta method from discoverynodes and related empty_list from discoverynode
add a comment
add snappy to compressionstrategyfactory and compressingstoretest
fix namespace for symbol
make wakelock final
remove lost code
fix quaity flaws
never strip trailing whitespaces in markdown documents
fix javadoc typo
made pgllistener constructor public
don t set a toast s view to null when enqueued several times
if access token is null do not include authorization header
fix upgradesstables help text
revert fix nonserializable createsessionrequest bean
fix typo in comment for provides annotation
fix buffer leak in bzip2encodertest
fix pgraphicsopengl surface field not getting initialized
add test case for updateusedbytesintiertest
fix malformed junit rule
fix reuse of data segment as bug reported by ford in ml
rename master server to meta
fix material exporting npe when the additionalrenderstate is null
don t move the fast scroll thumb with a list on overscroll
add disabled property to ehcachefactorybean
fix initialization cleanup in activity instrumentation test case
add back missing enable check correcting javadoc
add classifier parameter to the apklib goal to provide classifier
binddata should accept any collection not just list
changed chat app from fitcdemo to chatserver
fix the property name to respect the other related classloader properties declaration
do not play sounds or vibes for background users notifications
fix a test failure in localchannelregistrytest
remove call to rebuildwebtextview from handling selection change
remove debug print from test
fix date time format for gwt
handle cases when there is no value
appease findbugs with mkdirs return value
don t pollute logs with expected modal indexing start traces
add tostring to alert ie note class
set row heights for browse rows
fix a case where null was returned
add intersectrectangles to intersector
fix a bug where the intrinsic height was not updated
fix named based affinity when using ringbuffer
add method to get the count of roster items
remove input type flag from getglue comment box
add enotsock error code
add space for made name
hide password in exception messages of socksauthrequest
disabling breakpoint group worked only after the second click
make project structure navigation actions dumb aware
fix a string format error
add small sample to the javacompile task
add notification about change to internal state on keyvaluesource hasnext method javadoc
add the option to set transferstxtdefinesstationpaths option to use the transfers txt from the gtfs feed
remove redundant test comment
disable instance cache in minidfscluster
fix timing issue
add comments around case sensitive parts
add fix for case when bitmap createbitmap method returns the same object
fix issue with other month artifacts visible
add headers method to set multiple headers at once
add warning when javaagent isn t running
only report top peekingcard as visibile
made several internal methods of pyresolveutil private
parse parameters from multipart form data posts rather than just dropping them needed for windmill tests among other things
add safty to url detection in chat
fix xbreakpointmanager testserialize
fix problem finding location on different ip address length
add view tests for unknown schema
previous cleanup missed one string
remove redundant string allocation
changed the warning printed when freezing script test is found
add text constructor with default horizontalalign left
fix bug with services shutting down by itself
to catch assertionerror as well it could be thrown e g from org jboss as protocol mgmt activeoperationsupport
if we have an actionbar we do not want a translucentstatusbar
refactored svnutil getfilecontents reuse factory selection logic from svnvcs
fix correct userid parameter
add some javadoc
add unsetcomputedvalue method to graphtester
get rid of the codenarc output from the console
reduce logging from vectormapoperator matt mccline reviewed by siddharth seth
fix quality flaw
add getsystemphoneapplication to telecommmanager
remove unused imports
remove unnecessary calls
change message filter order
debugging too much abstraction
detect text or binary i o in pathlib path open
remove unused fileutils savetext
accept a manually entered proxy in the sip account reg wizard when the user navigates between the wizards pages
change f and s output type in charseq to
add ide error messages for result_type_mismatch
add new listener methods to the listener
stagger the compaction of rows
add ispoweroftwo check
make remove test code use the same api that the thrift interface does
mysql set editor support
improve java doc
remove star imports
renamed deletemappings to removemappings and their static equivalents for consistency
fredricho akifh fix a null pointer issue
fix wrong expandable state indicator orientation bug
remove unused variables
reset back the timeout
remove unnecesary conversion to long
docs add javadocs to eslogger and esloggerfactory
invoke interceptor from onopen
remove unsupported rewrite option from match query builder
improve abstract factory javadocs
remove debug logging
don t rely on lexer s null token start
abort accessibility node prefetch if parent node is null
corrected umatched version for template
add docs about null responses
updated assert message
fix for declaration resolve
add resolutionstrategy javadocs
don t skip null type in inference
reintroduced removealllisteners method since spring dm uses it
remove language level initialization hack not needed anymore
fix bug in aseq toarray object
fix force close from plain text browser
add debug message
mercurial tests missing status
fetch disallow_config_vpn for the current user
using correct bundle entry for align fields in column formatting setting
sourceformatter fix sorting
don t register a sync if the transaction is committed
piface binding reads wrong state for output pins upon startup
register allocate the roundrobinindex
fix quality flaw
restore wrong logger asserttrue contracts
add a debug message i found useful when trying to track down a websocket issue
remove unused imports
fix attributes generation for template languages
remove dead code
exclude specials from remaining episodes count
don t open many similar balloons when clicking on more in log
allow for null boltmsg when polling
permit mediaroutebutton to select system routes in toggle mode
add a space between recipients when doing tostring
appended some distinguishable text to the generic reason phrase returned by httpresponsestatus valueof int
adds xpath layout tests to dumprendertree ignoreresults list as android does not support xpath
add note about autosave loading in ondraw
add trace level for logginghandler
delete simperium db on startup before the initialization of simperium
remove unnecessary plain text tokenizers
add restcontroller to default server config
align default with context javadoc
allowed multiple spaces in group names
deprecate conflicts dialog without conflict elements
add pendingintents to one off notifications
also decrypt messages from untrusted sessions
refresh fs before searching for files on the local fs in python tests
fix formatting issue
fix resetting margins and borders on android
resize a popup window according to the preferred size of the component instead of its size
don t use flag_activity_single_top to start messagelist
updated methodwithgenerics test replacing map entry with entry this must work
silence the javacfiler warning compiler warn proc unclosed type files
peter full editor not set as current on first appearance fix
fix visibility issue in retrytests
integrate debug task to the execution system
fix httpresolution to follow entire location header uri including query
ignoring exceptions thrown from chained callbacks
make filesystemwatchertests thread safe
use more efficient entryset iterator instead of keyset get
use a larger timeout waiting for the distribution to be available
prevent pageup capture when popup visible
fix emulator boot
fix unbind failure through appropriate guard
add output for error messages
fix a shouldn t reach here error
strip underscore off gelf fields
fix crash in imageloader when feed had no image
add documentation of aar output to the build encyclopedia
enable new welcome screen for rm
add beta annotation
fix intercept adjustement according to set prior probability adjust all submodels
fix properties path regression in the build class
as per the future contract throw cancellationexception if the promise has been cancelled
log the number of lines read from file at the end of reading
add default value submitted by bill barker
skip collections for injection
revert commented out sysout on axis update
diff does not find any differences between databases
make nodeversion and npmversion obligatory add description to plugin properties
remove unused functions
fix remaining test cases
fix argument order in futures sample code
improved report for failing tests
set minimum size for transfer call dialog
remove unused import
updated randomboolean with chance
lost the import again
fix testcase remove sysout
change default work challenge background color
fix nfc docs breakage
fix test bug
fix fail while failing to load maps add on
fix compiler warning in xmpp action
performing a rebase
add database path print on server startup
fix local failing cxf tests
fix compilation issue in deprecated lwjglrenderer
avoid an anr under some conditions
don t show sidebar menu by default
fix authorization status message
ensuring file writable
fixes support lib breakage part of a multi part commit to move allowgeneratedreplies
remove unused intentfilter in locationmanager
announce final prompt again if recovering from a u turn when possible situation
remove javadoc for non existing parameter
createfromstream should create a bitmapdrawable which contains an instance of a bitmap
allow change documents during highlighting
add validator for cluster routing allocation disk boolean settings
add precipitation probability for wunderground
fix javadoc on counter histogram meter and time
updated class comments
add test for multiline string literal with semicolons parsing for postgresql
fix documentation issues with charmatcher matchesnoneof
agent fix pull code
draw smaller system status icons without scaling
deprecated npt ignore member add ignore method overload as the new preferred method
remove extra imports in blockservice handler integration test
properly initializes test dir for dumps
don t print httpd socket exceptions
improve logging and make sure we never times out live connections simple barrier
fix some tck issues
clarify what does and doesn t count as a lookup in cachestats
add permission requirements to sipaudiocall and sipmanager javadoc
cleaning up logging config
cancelling a future should clear the callback
always add the preview true url parameters to previewed posts and pages
use right exception class if no loaders registered
add comment about usage to wpwebviewactivity
properly cancel tether notification
refactoring remove unnecessary mergecheckresult not_exists_partly_merged
do not return subscriber connection to prevent getting events
don t scan target directories for annotations in multimodule project
fix regression when setting initial offset when docked
organised imports commented out unnecessary trace
set transfer encoding to chunked when response is gzip
do not store the widget type in lower case
remove unnecessary toast
add patch to the list of known http methods
grlightlocalvariable has a context
add more error info
fix parameter indexing problem
fix the build
actually send the scenarioresult to the ruby call
improve internalnioinputbuffer parseheaders
bug fix error import
add permitall to options on token endpoint
fix visible password issue on password unlock screen
remove debug logging on stderr
add logging statement for single test execution
git init action should preselect the directory from data context
disable checkstyle for kafka spout
resolve methods in buildconfig groovy
check npe condition
always set default model loader in glide load
datasource refresh fix
remove deprecated mws method
max write action for vfs refresh
optimize batchnode batch a bit
fix a bug where sync was sent before the first message
cancel test selections when they start if the client hasn t overridden the
corrected javadoc comment
delete obsolete import
better suppress wrong turn instructions but still one small bug remains
fix warning in topnoperator
add more intelligence to bluetooth connection state
fix crash on setmainsession
remove this prefix
fix condition precedence
improve comments and log messages
shadowspannablestringbuilder implements charsequence
read the current block before skipping
uses the testgraphdatabasefactory to create impermanentdb
fix initial scrollpane flag in scrollbartest scroll bar are not ontop
remove todo separate issue filed
add another constructor to pass the field name as well
expand message on incorrect spring version to suggest checking the classpath for unwanted jars
include post details when tracking submitted comments
add remotableviewmethod to textview setenabled
do not use unnecessary protected method
updated todo msg
remove unnecessary logging statements
sending drag start notifications to children add to view group
generatechangelog in maven now respects the outputdefaultcatalog and outputdefaultschema properties
after add buttons with proper index
remove leftover which was causing findbugs error
applied the second patch intended for the documentation with thanks to daniel gredler
add missing header
sourceformatter remove code no longer needed
fix the lock and the log errors
add liquibase should run check to springliquibase
fix crash that occured when loading a feeditem without an enclosure
use weak references to values in cache
hide progress window on background action
remove index comparison in danmakuutils compare
fix search field for windows configuration with black panel background
fix dump formatting of remotecontrolclient playback state dump
fix getstorageencryptionstatus in devicepolicymanager
only use flag_activity_clear_top on the search intent when from global search
catch decoding errors in commandresultcallback
also check android sdk path property
sql script execute restore original view fix
corrected a welcome url
fix breaking logic around getfilename call
add getters for idgraph s vertex and edge id factories useful when alternating idgraph with batchgraph
checkstyle have your upper l
fix failing test due to cache timeout property
correctly call listener on wakeupread if the source is complete
implement entertemp using another entertemp
featuredescriptor s setvalue method will not accept a null value
fix duration that broke with previous commit
use double math in des calculations
make encodingfilter enablefor type safe
make service initialize abstract
no need to optimize on the bool query parser we already do that in the optimizequery method
allow fishing success rate to be adjustable
fix sysui crash
fix test failure
ignored usermanager leak
remove all fabs in a floating action menu
add method to retrieve custom_list tags now included in comparison after parsing read menu
revert fix until root cause has been found
add log for failover
fix stupid typo in islegateeofundefinedelement
do return an instance for amazoniapmanager get
make empty string return empty string again not null anymore
add printstacktrace to throwable catch
fix focus for listpopupwithchildpopups
do not call isuptodate from edt
fix typo on presenterlifecyclelinkertest
remove unnessary comments
fix broken code sample
revert dismiss color fade when it is no longer used
fix some todo comments
loading rules on startup fix
fix bugs in the downloadobserver
allow different mention detection type when running on conll
enable debug and track pattern start cell progress
add type argument to pathrepresentation
handle alluxio tarballs with alluxio version inner directories
explicitly use image cmd
omit package name from builder class name
fix bug in terrain editor submitting new texture before scale causing error in shader
protect against null command map entries
fix identification of istag status during identification of the descriptor
set max res version if maxsdk version is present
fix merge error
fix nosuchfielderror for some devices when trying to convert dps to pixels
if there is no moon rise set today show the tomorrow
moved worker pool test to yieldingwaitstrategy
do not lock files
tidy up naming
add installed message at the end of a plugin installation
add a method to set a profile owner and active admin at once
replace custom aes crypto implementation with conceal as default
add some lifecycle logging to the daemon
move output panel help key registration to prevent npe
fix compiler warning
fix clearing linkproperties on disconnect
all jetreferenceexpression s should have a reference see jetreferenceexpression mainreference extension made jetpackagedirective not a jetreferenceexpression
remove unused getnumthreadsfornetwork
log don t display commit time if equal to author time
don t print the comment for disabled arguments
boolean type only has two values yet some people me tend not to choose proper one
disable sethasstableids again as it causes some weird behavior
fix exception message in restrequest
fix broken enter in inplace editors e g refactorings and live templates
ok button always disabled fix
add check for responseentity void with null return value
add javadoc to explain serial runner on grouppropertiestest
revert ignore logging tests that fail on ci
adds account jid information for the jingle nodes relay numbers logging
prevent nullpointerexception in getstringvalue if no value is set
add convenience methods to get the action of a touchevent
fix more javadoc issues
made handling of non right angles in getdirection return more relevant errors
na fix build failures
remember custom find in files searches
add infinispan storefeatures cluster mode comment
prefix remove unused imports
update the worker blocksize to match
make putjsonaa4 public needed for profiler
correctly handle zero length web sockets
restore new tx purposes correctly
remove non existent import
always create a new connection for http upgrade requests in the reverse proxy
avoid exceptions caused by writing vm options to protected file reverted
relax assertions on tests
cleaned up printlns
crank this up after implicit changes to revisionid
also add a good suggestion for languagetool all lowercase
set focus to scope chooser combo when selecting custom scope
make class as final
fix bug in multicast manager
disable the latest modifications
don t modify generator if there are no options
handle broken files in parse attempts better
fix go to related speed search
introduce method for creating shardrewriter
don t write lockout deadline if not needed
fix crash during activity transition with textureview
fix misnamed setter
fix checkstyle problems
add burt s patch implementing hashcode method in antpathrequestmatcher
do not merge change display state modification during dozing
remove depricated show method
add world getplayers
enhanced message for can t call public method of non public class
remove pointless test
changed dynamic suggestion height measure implementation
rename getprivatednsnames to getprivateipaddresses
add a constructor to htmldocumentimpl
remove warning in non eng builds
provide statistics via a sessionmanager method instead of forcing inheritance on implementatations
fix some tests
made fallback semaphore allow maximum concurrent fallbacks in unit test
android update np
fix the user agent test so it works with a different user agent string
improve builder javadoc
add message for assertion
taking owership of a typo
fix logic error remove multiple dead receivers
remove writing pin s to log
add logging statement to test to indicate effect of memory saving property
updated the flickrapi to use https
handle the failure of compilation better so we don t find out because of a classcastexception on a typeelement
vcs don t allow returning duplicate virtualfiles
fix getimplicitlockinterval check
fix branch detection problem in the current branch dialog
fix test when running under proxy
fix breaking while exceeding max intermediate rows
notify plugins of pause resume before queing js event no op
fix status bar bt accessibility message
add missing override annotation
fix npe when doing anonymous login
fix cs issues by remove unused certs
allow other scrollbar types
remove declared error that should not occur
remove system out
make editorcombobox completion for string comboboxes available during dumb mode
fix wrong expected exception in the test
allows fetch keys to fetch keys from system stores
working version of language selector
improved message when extractor is deleted
cleaned out modalitystate non_mmodal
fix comment typo
allow atn state optimization sam fix
check disposed state
updated the sponge permissions delegate for upstream api changes
fix memory leak in connectivity service when phone app crashes
clarified javadoc for searchtext string int boolean in solo and searcher
scan progress dialog can cause ui freezes
handle command line parse exceptions separately from all other in pinot admin
add bmp as image bmp
modified mock after merge
fix skylarknestedset tostring
remove weird import from previous commit
remove exceptionaction continue_wait
add baseline gradient computation workers
fix liquid id
making indent wider
switch off debug
retain signal level even when data is offline
restore blank lines in copyright license comment header
fix sbtree issue with distributes storage
remove old comment
add missing test
fix the delete mount point journaling
restore analyze stacktrace
fix vector3 nor is a bit more robust now
remove references to entityinterceptor which has been remove from spring data mapping
make unmodified field final
add pass code protection to authenticatoractivity
fix snackbar coloring of the action button
fix test for the moment in a totally broken way
wrap filewatcher startup with try
sourceformatter use contentutil get
use linkedset to preserve ordering of commandparams so we get the tag correctly
add missing support annotation
remove duplicated test method in nearcacheconfigtest
delete deprecated code
set nine patch padding initially to an empty rectangle
ask progress when loading recent revisions in revision selector
provide access to the filechannel under an image
sorry fix build again
add support for multiple requests in shadowabstracthttpclient
make eventcontentfilter s members final
add getapplicationcontext to the mock implementation
introduce better reset for when route should be recalculated to save some internet traffic
add constructor in order not to break open api
handling any exception that can happen on reconnect
remove unused constructor
remove flicker on initial drag from src position
fr set the french calendar in fr datecheckfilter
implemented server interface for base server
trying to fix windows test
don t show domain name of email based users
don t assign psi to this node
fix bug where all modules in the new moduledescriptorcache were flagged as changing
fix missing range in streammessagecount
fix ensurenodesareavailable s error message
don t show private notifications if a secure lockscreen is merely occluded
test the camcorderprofile before trying it
add missing bindings in guice integration
jeka propagate internal jdi exceptions to ui
fix syntax on endpoint for component docs in camel chronicele
forgot to revert the experimental change
remove accidentely committed line fixes failing tests
remove an unused variable from audioservice
fix eclipse neon nag
fix issue with attribute resolution
fix boolean option enry reset
fix bug of userincludeplugin
fix comment for floatingpointbitsconverterutil
show the reason in ui rather than popping up idea internal error dialog
only overwrite the data point ttl with the configured ttl if a data
updating rest index query to show fuzzy search and multiple keys
setting the hazelcast s shutdownhookthread s name to hz shutdownthread
fix restore maximized state under linux
fix inappropriate suppresswarnings
update on hibernate modules for max size policy impl
add a todo
fix signature test failures in javax servlet jsp package
do not update skeletons for detected sdk twice from show more
throw a exception on unused methods
add command to replace duplicates
add error checks to ensure switchtobubbleview is not called when there are no active tabs
make output a bit more verbose
remove merge leftovers
key on exception class not object itself
new titled border no offset
add missed te check for client gui close
fix tiny typo in test name
fix qt zoom layer
remove a system err println from a test method
fix cut and paste error in logging of ignored columns
remove redundant null check
improved exception message
fix sysui crash when no metadata is provided
changed configpattermatcher to static
hidden some methods of resolveimportutil for python
fix a bug where custom notifications were invisible
fix the build
add a default flush interval for hivebolt
remove unnecessary code
fix builder names in messages
put the marker on the server ml file too
fix selection unselection when using badges
remove redundant and confusing else
remove a nonsense comment from tokenizer java leftover from earlier patch iteration
add documentation in iterables getnext iterable object that says that if no default value is desired to use iterable iterator
method made public to fix kotlin plugin
do not require an existing node to start the spider
do not allocate arrays of zero length
add xml validation for sar subsystem
fix npe from garbagecollectionmonitor on jdks that don t support performance monitoring
fix some general silliness
fix unit test androidresourceruletest
made compareto method simpler improved routing speed a lot
add the extra data definition for append location in the view request to the browser
don t switch logo if not needed
fix incorrect comment in quicksort java
add spring websocket support
fix weighted mean wasn t reducing the sum of weights
require save before preview of shinydoc
add missing test
remove a todo that didn t actually show big improvement
change websites to urls
add junit to test time dependencies in grailscoredependencies
use textrange to show execute file icon
add check hastxstatewithchanges when creating an unique constraint
invoke disconnection handlers on timeout even if we didn t successfully connect yet
add sensormanager to service bindings
allow xmljson strees test to run longer on slow ci servers
ignored memcachetest testbulksetget_withmanykeys due to failures
fix a typo in recyclerview javadoc
add some comments to explain some cryptography related classes and to dos
remove unneed initialization
add junit for gbm on single predictor column binomial test passes
make views array in viewgroups optional
suggest name as classname dot methodname
updated study project name
remove member list update setting in migration tests
keep keyboard focus on window if null
restrict force echo behavior to rnw and rmd files
remove dead code
never return a proxy
replace stacktrace prints
fix crash in test of files downloading uploading
changing preconditions order may have significant performance implications
fix issue with custom filter
test throw correct error within assertbusy in truncatetranslogit
fix debug tool tests in case of csv suite
add commandresult isok
remove some deprecation warnings
remove unused variable
can produce bad code with array initializers
eliminated unwanted method
how bout a line break
fix a bug where the text directionality was inconsistent
remove minmaxviewport names
fix failing test
properly encode folder name in messageprovider delete uri
remove repeat from test
unreverting parts of console
fix unstable tests
elasticsearch restrict visibility of defaults
copy premultipliedalpha in copy constructor
fix incorrect html excape in javadoc code tag
fix incorrect serializable
fix typo and other errors in tieredblockstore java
add back checkcast before invokevirtual calls
fix behavior of receiver checkbox
add support for logging when cas tries to determine the principal id for a given service
fix npe when internode compression is off
catch everything when deserializing and extracting
add bytestring asbytebuffer
remove unused variable
remove unused import
fix test failure in testresourceschemacollection
disable log tx by default
improve literal serialization
update eh equals to work with any object
add the missing java doc for the new public method in testsupport
remove bytecode storing from tests
add dimension service key to fetch dialog
tolerate null char for string entries in stringcollector
recover actor push
xml config properly initialize ws message broker stats
improve upload error logging by add a message so that the full stack trace goes into the logs
remove obsolete lines
generated example code with current lib version
reenabled javaprint method
remove unused import
improve error message when stress is having gc issues ninja
rename pendingcount to activecalls in hivesessionimpl for easier understanding
changed visibility of orientedge converttodocument to public to be invoked by create edge sql command
remove unsupportedoperation special handling
prepare data in project import builder for import
fix keep alive broken by refactoring
don t preserve the window on client side when pinning activity
add sourcehttpmessageconverter as a default
don t bother printing deprecation warnings in whitelists since they wouldn t be promoted to errors
stream closed exceptions in logs
accept docroot in parameter documentation
add null check for session_timeout_ms
remove checking for result_canceled from onactivityresult
stop the activitymonitor when calling finishopenedactivities
remove unnecessary whitespace
add yaml as dependency for rsconnect
class cast exception
handle all camera api exceptions during init
add comment and convenience method for getboolean
made getserver public instead of setserver which imho is an implementation detail
moved warning logic down so that no more tickets are isues if the mod is over it s alotment
remove unused import warning
use of delegate would cause memory leaks in eclipse
remove unused setting
ignoring proxy tests
fix build breakage
insulate systemui from ids in inflated notification views
add subject feature items
pass button index into inflatebutton
skip gc collection notifications for collectors that run mostly in parallel to the application
optimization do not create roster item when receiving an unsubscription request from an unknwon user
poke the spout telporter so the correct packets are sent
only configure magic link ui when initial start up when savedinstancestate null
settings search improvements ak
avoid memory leak in sample
exclude input_method_manager__served_view on m
changed weights from ints to doubles
add getter for executetime
fix npe in getting a user s fork if repository caching is disabled
provide an array with all languages except the demo language
delete no longer relevant comments about url filter
just use null
dumb aware generate action
put rbm activation back in
don t create the src directory by default
add internet acess to for tourists group and also index them when generating the map
add comments to tachyon ls method
optimized camel jetty writing response
allow the support library to work directly on gingerbread
eliminate eclipse warning for unused commands
add more comment about the frame position add braces to the if statements
no longer add a new account twice to the list of ordered accounts
delete hook should inherit dumb awareness
add hack to natives extraction to extract
prevent notification swipes from getting canceled too soon
fix a bug where subtasks data could become corrupt after migration
remove part of test
fix spelling of intiialization to initialization in comment
also use the parent constructor here
en move rule so comment specific to english makes more sense
add necessary space
compare paths sensitively when remove common parents
allow retrieving of gpuimage used in gpuimageview
return different grammaticalstructure subclasses based on language
make sure we times out on connect operation when the connect operation occurs on the calling thread
hide pseudo columns from autocomplete
add title for github based projects in webstorm
remove dead code
changing array indexing up arraymultapp
enable the home button for traktsyncactivity also set the title
fix some warnings
try a different combination of ignored tests
fix bug in timestamps micro sleepuntil
allow subclasses to override the base directory rather than hard coding to current directory
throw error if cardinality aggregator has a sub aggregation
hide thumbnail strip for photo cards
correcting jsonimporter to use the correct parser
remove erroneous checked exception declaration
revert previous fix
use new setter methods
no bts resolve pinpoint build fail
fix wrong package of resource references to bpmn process moved directory due to strange svn problems
add null check
suppress warnings in mappingjackson2xmlhttpmessageconvertertests
activated unit test for cogroupoperator
fix processor compile issue
enforce permissions on start stoplocktaskmodeoncurrent
don t allow duplicates in the author field of git commit dialog
fix bad super call in phonestatusbarview accessibility
fix method sorting in annotationdirectorypool
fix error overwriting when restoring error
clarified javadoc for isradiobuttonchecked in solo and robotiumutils
fix missing space in log message
fix bug in cloneforspatial use in spatial
move focus to visible behind activity when we move the home task back
do not restart daemon on cache mode change
fix typo thanks to benjamin hanzelmann
enable required dependencies
update the status in the selector box on the left of the send button
put all debugging onto a single line
add missing javadoc
remove context dependency from squidutilities
fix error message in readresponse
this broke reactnative
remove dead code
includes the current stack in the exception not just the exception from where the db was shutdown
add tab listener for feature toggles
add wait for green to prevent test failure
resolve merge errors
disable multipolygon test pending discussion
stop listening for wifi interface status changes
set exception cause when re raising ioexception
changed timeout time for two finger tap
add new watch doesn t start editing fix
enhanced textrunnertest to catch missing return values
temporarily disable multiuser for specific builds
reduced getparametertype call not supported log message to single line
add jndi permissions to remote naming test case
make the bean actually implement the sayhello method
add extra validation for invalid paths
remove deprecated methods cleanup idetalk idetalk must use our built in web server will be done in next commits
log fix hidden graph because of the date filter existance
add abstract avoiding behavior to the javadoc
add test for compresstype with bounded wildcard generic arguments
fix advice correction
fix leftover logcat tag
boo i missed the private field in the last commit
fix some copy pasted javadoc in streamsconfig java
fix is modified check in dependencies configurable
fix a bug in parallelmodifier being only as long as the shortest modifier instead of the longest
updated the copyright notice to current year
fix potential threading issue if fluidregistry loopupfluidforblock is called from two threads at the same time before being setup
modify activethreadcounter log level
fix the eclipse warning of the assertionclause
add runtimeservice startprocessinstancebymessage string string
add param newline
log blog url to assist debugging
fix issue on remote exporting of databases
don t clobber intent flags when starting recent activities anew
smallfix remove redundant initializer in datafilechannel
simplify ignore whitespace comparison and fix npe warning
delete the todo to try creating an abstract superclass that networkbuilder and graphbuilder could derive from
fix npe when trying to add a new handler to the map of directed presences
remove unused pref value
fix z order issue child item camera android
change headers builder to be final
add copyright notice to downloadlistener
vcs log avoid npe on dispose and better disposing order
add todo for source of most of the current ci breakages for parallel execution
fix iscontact binding
fix builder inconsistently respecting oredict
add additional debug info to upgrade test
remove top level filter parameter from search api
fix another typo on docs
fix api jar temporary classes directory classes can potentially be mixed up
remove dead logging code
leave filter in place
add the version number to the json multicast response
fix up whitespace
fix crashes of int string default application mode which resets that setting
fix boxconsumer compile issue
solo getview class t should return t
fix the constructor comments with correct parameter in defaultmemorymanager
remove obsolete code
deleting old make implementation
allow bytes as well
fix flashlight availability
changed the class api to expose the api domain so other classes extensions can use it
changing the log level from info to debug
check that vms match if class loader is null otherwise we may take last result from the different debug session
catch both filenotfound nosuchfileexception
optimize loop in orctester
remove checkstyle suppression add javadoc
printed class short name as alias in command info class
fix doc typo and remove redundant imports
bump packrat dependency to latest cran version
fix permission check in hasenrolledfingerprints
disable call log for users with disallow_outgoing_calls restriction
update usage information
fix issue on browsing of entire cluster class
project service seems a bit late a place to initialize stub element types
resize lookup if new items are add slowly
fix problem that hue binding always sent two requests to bridge for color commands
reset main tvview only for explicit reset call
remove unnecessary suppresswarnings
add compile fail test if annotated deeplink in non static method
remove the other declarations of ioexception which bubbled up and javadocs indicating such
add license block
make textareapainter load tab size from preferences
handle undo in textview fields with no cursor
fix poi not found
add url to the command line interface as more info
ignore idea internal files
fix sessionname param in clientapimain
fix cube map shader crash with multiple lights
used the supplied one
add kamil kaczmarczyk s full name
fix ignored exception
oops missed this use of the mapfactory
rearranged the classes in the segmenttree
correct the name of mysql s xadatasource implementation
to avoid problems with too long filenames use short name of class containing hang as directory name
add value comparison to test
fix npe on version checking
use shorter uuid
add a bit more info to constant field javadocs
add methods to get at the sherlock version of the activity
websocketclient leaks file handles when exceptions are thrown from open
also call postwrite in vec set
remove unused block
select the tab on startup for easier debugging
suppress turn prompt also for recalculated still opposite route
handle chunking where the line is terminated with n rather than r n
fix file save
fix xpath implementation throwing exception
only maximize console if it isn t already maximized
add flush logging
update kryo class path
add builder withselecteditem with identifier
make streamcharbuffer extend groovyobjectsupport
sort suggested plugins by name
add a bunch of useful info to the debug output
erased commented code
fix teamcity violation
do not reset watched flag if failing to load from trakt
add compaction marker to in memory list of components
fixes cloned notification actions missing icon
improve error logging in installcommand
merging order fix to highlight symbols correctly
make paging touch slop smaller
display full title in item screen
add deleted tempfile setup
avoid outputting a deprecated attribute when marshalling the xml
fix bug on query with between operator and dates
attached parent ctx to grails application object
fix a few typos
reverted inadvertent api change to solo regarding generic activity class declaration for getcurrentactivity parameter
use arraylist instead of linkedlist
try loading the class first using the atmosphereframework
remove unnecessary comments on castmanager
fix quality flaw
fix bug by luca molino on new http upload file command
ignore eof exception while writing data to client
fix class name
add missing import
fix javadoc typos in consumerrebalancelistener
remove wrong import
fix uidmanager to pull the table name from the config class instead of only
run new visiblegraphimpl
add undemote_method to contactscontract
fix compilation error
fix source formatter
library fix logtag for twspannablegridlayoutmanager
add check to the equalswithdelta matcher
delete unneeded pong logging
save a few cycles
do not highlight classes with main methods as unused
add exception handling for xml errors
remove magic number
fix issue with nested exception during tear down causing initial exception to be forgotten
ask the user if he wants to rename the old style property itself or its getter setter function
remove final from template fields
increased performance test expected execution time in order to make test pass on build agents
execute cassandraartifactcache stores asynchronously
fix analyzing nested scalar and aggregate functions
fix code issue repeating startswith when it meant to have a startswith and an endswith
index update error log fix
remove unnecessary semicolon
add note about httpconfigurable config storages
simplify check for street edges
fix nullpointerexception in the contact list
add helper method for matching type which returns default if null
deprecated momentum on player remove tomorrow add velocity
fix icon in action mode for favorites tree fragment
add field for max number of fields without scrollbar
fix linear decay def make exponential the default
fix casting bug
revert change on wherecond name
fix non handling of html annotations book info
ignore unknown fields when deserializing job object
add log messages to pagefetcher
close the open files before trying to move them
add missing test for calcsilhouettecoefficient
allow settings to set background data restrict on secondary users
fix getactiveapntypes for gsmdct
don t break bc store classes dex in smali
throw away transcriptscreen backing store on session finish
add comment to document this weird casting technique
fix bug and support t f for boolean
display localized message when flagging movie seen unseen on trakt
fix merge bug
always return true from toolbaractionbar onkeyshortcut
fix a bug where the expand button was invisible
remove localfilesystem getinstance dependency
remove todo from the source code
fix proper comment
fix erroneous illegalstateexception
changing string to wordpress com
do not check repository unless it is non null in gethistory
revert fix issue on shutdown while a parallel query is running
disable suppress for parameter for lambda without formal types
fix bug that caused scroll position to be lost in reader post list after returning from post detail
make registerrequest public so that rest api servlets can be add from
fredricho add desktop user agent to theme preview feature
fix sip tests
fix missing resourceleak close in abstractdnsmessage
changed javadoc regarding backpressure
revert kl s change requesting a ore will register it
detector regexp for windows had a pointless grouping in it
fix local file dir
now using proper android instead of java logging in socketutils
remove remaining usages of idea_marker_class
fix npe when passing null tvcontentrating
implement support for new second level media window type
remove operationthreaded setter from explainrequestbuilder already available in base class singleshardoperationrequestbuilder
type the getter for the adapters
revert break test
add new line char
include better error checks for uri to file path utility method
only have magic link callback trigger if error is taken
remove deprecated methods
remove interpolator import
remove extra spacing
relaxed performance test condition
makes sure guava only uses public gwt apis
add unsubscribe to abstractviewmodel
temporary comment bad code
improve input filtering of the markup we get from the atom feed
use correct snapshot build number
fix inconsistent state after forced removetoken
initialize the bundlecontext before starting growl was causing an exception when trying to access resources string at startup
prevent spin when response stream gets into wrong state
don t clear the pattern when window focus changes
fix bug in rectangularshape setbasesize
remove identical branch
fix typo in error message
make the renewal last update timestamp volatile so that the expiration task does not have a race condition
fix codestyle violation
renamed method to prevent same naming
bind lifeline only to localhost
test make parent breaker check less strict
work around ci build failures
remove verbose logging statements
fix a couple of bugs
fix test compilation
make apns java final
make mongodb hostinfo serverstatus and dbstats optional
remove unnecessary suppresswarnings
avoid race condition double check locking
noticed a few typos in the logs
fix precondition string in configelement
fix the issue of selection will reduce number of rows more than size
who killed sprite getrotation
remove unused code
remove some todos in mapinterfacetest
shortened output message for smooth
don t show too much output when console cyclic buffer is disabled
remove the unused ril constant
implement now applicationwindow interface
remove forced refresh on settings update
clarified javadoc for getcurrenttextviews in solo and viewfetcher
fix utilstest on windows
only get output size hash when building locally
remove unnecessary return type for method
sort the collection of countries by number of views
remove the leaked window workaround
remove double up unregistering mpackagebroadcastreceiver
add classreflection isarray class
add getlegacyvpninfoprivileged method
throw exception if via count is uninitialized
add more info to view getlayoutparams
use a subtarget to avoid contaminating the container
using isnullorempty method
fix documentation of parameter
add new beanserializer option
add countrycode to fakeaddress
fk create fix quote source column names
add issafe to fqnameunsafe
remove usages of deprecated callback adapter
remove unused import exception
remove unused code
ignored failing test
enable dan s new resync accounts on reconnect code for push
fix bug in setting maxwaitqueuesize
bump maxmessagesize for android testing
add a reset to default value on a remote network test
use projectcoreutil isprojectorworkspacefile
add missing license headers
remove unused method
add missing classrule annotation
remove unused imports
remove unused code
prefer editor apply over commit
fix debugging message
always explicitly use a locale
do not create intents for application xhtml xml
add a check for misspelled span change
fix broken event receiving firehose
avoid magic number
fix missing trailing in resource paths
wrap class cast exceptions
remove notimplemented exception
disable the pagecachetest backgroundthreadsmustgracefullyshutdown
add new line
add tostring method to enable string rep of results
moved maxdb support to extension
apply review comments
fix focus issues on rv
updating ioutils header
fix a warning
add some useful imports to mvc scripts
fix item selection npe
fix bombed test
plus icon color getter and setter
use different names for mockjdk and fulljdk
don t collect property accessors twice
fix rounding resulting in choppy timing animations
add autotest to some examples
fix checkstyle issues
take padding into account when clipping image
fix test failure
remove duplicated code
improve icon drawing
throw notconnectedexception on login
library remove logging leftover
fix non spoken instructions
avoid potential npe
textchange transition now uses charsequence instead of string
fix wrong variable name
do not track analyticstracker stat reader_article_opened here since we re counting the first article twice
remove notnull annotation that was imported from com sun istack iternal which triggers a compilation failure on openjdk
don t create a session manager unless needed
recompute mslideoffset in onlayout to avoid rounding errors
add detachvolumes to strategy list
don t log warning when stopping container by name
add test for failing compilation
using x_downloaded_full flag instead of the x_downloaded_partial on to determine if we need to show the downloading icon
add missing nullable annotations
add support for using a transaction to perform batch tasks
remove unneeded import
properly ignore peergrouptest peerpriority
in jsf tag attributes must generally be deferred value expressions but idea does not flag as error immediate evaluation expressions
missed a document me
remove the spurious logging of classpath file names in classpath
remove some necessary javadoc
send installeruid for verification when using packageinstaller
missing one xaxis entry
remove weird formatting on a comment
when disconnecting from an smtp server send a quit command to be a
source formatter exclude bin eclipse and classes compiled dirs
hide waiting list if empty
remove leftover useless code
add trace action search type to geoboundstests
got rid of a compiler warning
do not cache archive content unless asked
fix removetarget to remove by id instead of index
fix an issue introduced in an earlier commit reordered the arguments in one method for some reason
smallfix remove redundant intializers in filesystemblocklocationintegrationtest
add code to dispose texture mesh and sprite batch in waterripples example used for lwp
keep selected positions and checked items in sync
notnull just to make sure
remove ruby residual
use the interface
deprecate a method not used anymore
smallfix fix comments
do not consider the names of root directories when using pathsensitivity relative_with_file_names
request the focus in contact list when logged in
add additional check for proxy filter by address
print wallet balance and receive address earlier
don t match network requests to legacy api requests
correct a typo
fix bug in uniqueness determination in optimizer node
completed typepool read of class without annotations
remove debug messages
fix import order checkstyle
fix typo in error message when the same file is in both srcs and hdrs
fix a problem where shellserverextension assumed the shell was enabled during shutdown
using different seed in foldleft foldright tests to show the difference clearly
add test timeouts
fix bug in parseandmarshalmodelstestcase
fix wrong javadoc
fix config option loading for titanfactory builder
add test for when imagetagfactory has unset params
revert restore lost node start no op check
remove obsolete todo
keep track of sheetview s height
fix tablayout s flickering for good
rename water api register registerresourceroots to disambiguate multiple register classes in the codebase
fix compile error
remove unnecessary code
corrected the config javadoc
fix null pointer
move isbikeparkandride to osmwithtags
add a warning about the ssid format in captiveportaltracker
fix potential npe when one of the collaborators only has been collected ci fix
fix invalid traffic class in test
handle unsupported protocol errors in webview better
add method java isjavafile java io file
fix javadoc wrong for splitter dsl
sourceformatter rollback source formatting
create the jobmanager after booting
fix a bug with updating extras in a pendingintent with an intent stack
don t pretty print nan time
add javadoc comment
log remove not entirely correct notnull from fields initialized in setup
add null check to embeddedallowed check
avoid duplicates at editor folding settings
avoid npe in printspooler when changing language
remove some dead code
fix npe and hide progress bar without media
make constant field final and completeable
java coverage runner should accept java based configurations only
do not access the edt in daemon mode
log the version
handle cases where bike walking starts mid block
add support for canonical class names in mutableexception
amended error message
fix trigger sensor lifecycle
make threadcpustats a daemon thread
add webrefs for loadxml and loadtable for reference doc generation
remove dead code
add missing javadoc
remove extra debug
fix the cs errors
add boilerplate for nativeexecuteforcursorwindow call
fix error when printing domain values of timestamp with time zone type
fix copyright message
add isfinishing check before showing toast on failure
accept rubymine installation home
improved javadoc for metaobjectprotocol getmetamethod
make a wrapper
fix divider before text part with filename
fix typo in stats computation
make rootmodel public
fix a problem where the local transport does not fire a writecomplete event
fix build break due to additional arg in bitmap ctor
test use a large threshold to prevent relocations in recoverybackwardscompatibilitytests
don t allow screen invalidations to go outside bounds of screen
excel import sometimes drops last row of data
fix a build break
add baseactivity toastonuithread convenience method
fix scan and query operations to honor table name override
fixes a problem with the rtp port number being used for rtcp
add optional outputbasetime for log writers that wish to explicitly specify a base time
fix name of a test data class
propagate blockstoreexception in blockchain setchainhead
catched more ioexceptions to print them in debug
consider modules without goroot as non go modules
do not import the callback explicitly
you can now register beans that reference grails runtime beans in a applicationcontext afterconfset
fix npe on ee concurrent contextservice s transaction setup provider
actually fix formatting in sketch java
add gethighlightedfield and hashighlightedfields to messageresult
lost deprecated highlighting for fields
revert running im thread with bg priority
fix debug log
add spoof bit on second packet missed in previous commit
make beanpath casts threadsafe
add tostring to torodocument
instruct the inspection that int char conversion is correct
simplify visualtraversevisitor sleepafter
do not commit document explicitly delay indent detection until document is committed
add fallback_missing to hystrix rx netty metrics stream
add method to close a transaction with a pstmt argument
vault uses static inflater and deflater fix possible race condition when working with two or more projects
add replicaset annotation to test in driver compat
fix last commit
do not hide input output
check regexp intention action language detection fix
check if gradle module has kotlin runtime classes in module classpath
datasource rename fix
allow prep in rrc
tapping outside of main menu popover dismisses it
only include name with an address if it s non empty
remove wrong urls
improved saml webview rendering for non responsive idps
make key for caches a bit more unique
also overload execute without arguments to prevent compiler warnings when parametert is a generic type
logged exception in http service
add isdestroyed method to match framework api
fix error in afcclienttest testwritespeed
add trace logging
restore progress on directories
add nullness check
remove unnecessary copying of headers in genericmessage
fix trivial typo spotted by benjamin bentmann
fix compilation error with javac
no longer using deprecated methods of filtercontext
remove extra spaces after deleted statements
log better progress indicator text
package requirements and skeletons tests should use fixture from test task as it is recreated every run
delete the orphan blocks while initialize workerstorage
flip textureview s layer after calling setsurfacetexture
add missing suspendable
r maxim mossienko rollback re setting thread locals for release
improve javadoc to mention file names
remove duplicated collecting data text value node
fix building artifacts on windows
reporting unexpected exception as compilation error
fix some minor javadoc issues
fix error message to match code
remove duplicate fixture setuserdata
add convenience methods to register a proto compile action
incorporate the null check from the previous commit into the containing if statement
stops bad behavior
avoid error for negative lat lon
fix error msg loading rackdc properties when system property is specified
compare result timestamp based on millis utcoffset
fix workspace file existence hermeticity
tag retrofit request logs with error category
add a pair of methods to help optimize countingmessagestream
clarify peereventlistener javadocs a bit
suppress a newapi lint error
correcting path for magic link login api
do not show red error bar for suite that contains only passed and pending tests
clarified javadoc for pressspinneritem in solo and presser
remove unused stubbing
missing offset when emulating lookup item insertion
reset statusreceived when retrying connection
reuse the mac instance just constructed for bit length calculation
expand timeouts on jdbcextractionnamespacetest
avoiding creation of toplevel tmp through tests
remove noisy logging
fix modifier order in org apache naming
fix missing cross ref links in the mediaplayer java javadoc
include a maps value type in droptypestatement s isusedby
mute this test which is wrong and aggravated by the logging change
awaken listview scroll bars for overscroll
fix checkstyle warning
fix a crash when dumping groupmanager
clarify documentation for webview loaddata
add accepted values for setting status bar visibility
throw the sorting exception
fix versioned equals remove useless null check of this
resolved some findbugs issues
fix code formatting and comments
add illegal_sim_or_me error code
used an old parameter name
do not wrap runtimeexception
recognize xts objects as data in environment pane
add more debug tracing
fix for backpressureutils method javadoc
fix context inpredicate not working underneath testnotmatchers
make cancelling a timer idempotent
used string from resouces
fix incorrect param names being uploaded w o prefix
introduce constant add javadoc
do not skip empty lines in copyrights
wrap log statement with logger isdebugenabled
prevent premature window replacement
remove chatty log message
change the path to prevent tests from faling each other
remove repeated allowmissinginputs module check
remove android imports
put the this_reference_in_arrowfunc_of_objlit warning into lint checks
pass a defensive copy of dhcpresults
tidying string to char conversion
fix getmatchingparams if tie was found before best match
fix sql phrase
add note to nearcachetest testcachelocalentries
remove code duplication
remove stale test
avoid image disappearance after upload
encapsulate train wreck code
only apply session fixation protection strategy if request isrequestedsessionidvalid returns true
mispelling javadoc in the picasso class
store the current country in call log
minor testing tweaks
delombok now prints a space in between method type args and return type
avoid logging optional
fix ajp test issue
invalidate structure when adapter is changed
append the test case name in the beginning
make execcontroller getnumthreads return the right value
make minute standard histogram interval
rework for code style consistency
fix link tag
findwordundercursor didn t include last char of word
remove test code
allow top sleeping state in app process verification
clarified javadoc for solo click long onview methods and corresponding methods in clicker
fix check for svg file if header is missing
add license header
add test for multiline scripts
fix graphstore call for new table observer
don t move stack to front when creating a task for activity to be started
remove unused code
customview view will use themed context for inflation too
create volume help
reduced visibility of aggregationoptions builder constructor to private
fix issue with sub classes on database importing
fix the remaining test failures
fix compiler error
add translucentnavigationenabled getter setter
ignored topicoverloadabstracttest whenblock_whennospace
eliminate stray println
confusing search results for existing template
dont add new muc to roster
fix default marshmallow pressed selector
improve invalid instant error message
fix non deterministic pivot test
avoid duplicates of external jars
remove unused packages
disable fix all action if only empty quickfix is available
documented cache block
add message on non availble attachment provider
fix parsing bug for others
don t show hint file not found as error
cleaned up uidialogcreatenewworld java file
vertex creation reduce nn ipcs gopal v reviewed by sergey shelukhin
remove unused acquireincrement
remove unused fields i sstable in simpleslicereader
scroll pane resets border on l f
fix a dupe bug with stripes pipes
turn on model description validation
increase allowed time for parser
fix typo on pattern for i
assign shortcut to the copy content to actions
remove unused import in text container deserializer helper
remove print statements from docgen test
fix broken tests
fix edittext rendernode invalidation bugs
remove copy pasted value attribute in mongo
fix reversed precondition that broke aliases
implemented updated astyanaxcontext methods
remove logout listeners
properly handle top padding in staggeredgridview while freshly populating items
fix crash in log message
remove bogus pmd warning initcause is uesd to preserve stack trace
add a convenience method
revert unnecessary name change
ampersands have problems launching via start on windows as well
fix bug in hashslotarraybase
print a log warning if lock time is set and a sequence number has not been
clo needed main to generate man page
changed the loop increment value to final value
update default task info refresh interval
fix a comparison
on sign in put indeterminite spinning progress meter in actionbar
fix selected algorithm
add option to create frames with positive response for regression problems only
make phonewindow aware of layout direction
inactive floating toolwindow doesnt request focus on show
avoid supported abi list containing unknown
fix bug on passed parameters
fix plugin argument signature for markdown plugin
fix integration tests
remove trailing comma in initializer fails on some javac versions
hide gitignore selection if there are no gitignore resources
runtime properly set cause of producerfailedexception
capitalize words in the mp event name
implement consistent ruler label night appearance
catch throwable noclassdeffounderror and don t abort loading it s probably a badly named class file
incorporated pr comments
do not start external ann pass on no change
change subscription to compute effectivedate from eot policy differently when ctd exists but is in the past
fix task executions view first table column background
adjust _numoffsets when filtering expanded columns
test remove global scope mention from elasticsearchintegrationtest buildtestcluster
throw an exception when host is not found
remove stack trace when logging to ga
fix english typo in voldemortbuildandpushjob
enabled partitiontrain in machinereading
tweak bulk loader config handling
do not log invalidrecipeexception
fix npe with inpurgeable bitmaps in getallocationbytecount
fix mongo storage provider name
add a small delay that is needed for some special cases
fix forest generator
add regions for include code
add tostring to carrieridentifier
delete dead code
fix javadoc build
fix preferredproperties translation when partitioning on empty set
fix minor issues using filters programatically
suppress deprecation warning in statusresultmatchers
prevent unexpected scenario
checkcanceled in python resolve
ringtone pref is not just simple clickable preference
replaced the call to deprecated method
add lexeratnsimulator gettracestream
add api to replace unicoded digits to ascii numbers
add setter for usepooledmemory parameter
oauthupdate now supports pin
fix handling of shortcuts with second stroke for local actions
remove unused logger
remove unused imports
fix exception when writing to cache with an uncached service
could not save application settings
add a key for setting carrier s cdma roaming mode
do not dispose lighttest project if light test was not run
expect errors on importing from default packet
verify media type in discoveryjerseyprovider
increased time for hot file indent detection
unbreak the compile
properly escape templates
commit and compact the database on startup
remove twitter license headers
raise visibilty of types to public in request classes
fix a bug
replace throwable with unimplemented
create lineextensioninfo from textattributes
add missing constructor
ignored everything after the first exception rather than the last exception when checking for stack traces
remove redundant abstract methods declarations
fix daylight saving offset in astroconfig logging
fix typo in javadoc for typeevaluator
add note to jdocs that this exception should never get serialzed
remove unused attribute
remove unused field
temporarily ignore new client tests
print the keep going report via the superconsole api rather than stderr
fix regression see comment
clarified time unit on the timeout set methods
remove text watcher
fix jitter for shard compaction discovery
fix the broken config argument name when launching the application in redeploy mode
changed check if controller return type is result to be child from result to allow extending of result
remove unnecessary pattern instance
jdbctokenrepositoryimpl updatetoken should use lastused arg
trimming classes output
fix a bug where editing task from widget wouldn t have required id param in intent
don t use the classpath to look for gsp files which are no longer packaged anyway
fix npe in svnconfiguration initialize inner state to default value
remove unnecessary parameter
add flynx and uc browser to the whitelist
make internal methods package private
crossoriginfilter javadoc says exposeheaders but should be exposedheaders
compare types by equals
fix cloning to not confuse the hardware skinning safety check
fix null pointer exception for malformed feed comments
add the encryption option to pptpprofile parcel
layouts simplify get width height used in baselayoutmanager
commiting missing imports
remove granularity check in doc collectors since the unassignedshardcollector is also a doc collector
avoid same class to be present simultaneously in the same package
smallfix add openatposition for s3underfilesystem
fix view restore crop window may be outside image bounds
fix null body header after transform called bean that throws exception
revert preventing indexoutofboundsexception if search result is empty
fix version description
fix unit tests
remove unnecessary methods from tuple
fix bug when add set_install_location to settingsprovider database upgrade path
remove redundant cast in exceptions
now the massive insert intent reset the old settings
fix compilation error
made delay an ifn
remove todo that will never be done
fix bug on drop database for remote connections
improve generics usage in paginatedrepository
not covered message
disable lexing optimization correction check
add more javadocs
deactivate the navbar button for my tickets until it is finalized
add a static helper method to access the current running main
add one redundant for new code null check back
git log support extended regexps
remove deprecetad settings in graphapi settings
test can not be run on ibm jdk
fetch commandline from processhandler instance if possible
fix test for new c1nchunk compression
optimize the code
resolve securityexception on loadcustomlabel
enforce permission for changeapplicationstate function
only send extra_stack for private intent
add a distinct clause so we only have users once
convert test to pylightfixturetestcase
don t add covariant implementation for bridge methods
properly closing the stream and eventually remove it when receiving data frames
set content type to plain text for accesstoken in oauth server mode
add javadoc for variable masyncjournalwriter
changed member variable to local variable
backing out an errant change
revert return property with unknown calls for target expression stubs with assigned calls to property
suppress screen lock sound effect on first boot
synchronize creation of testprogressevent
replace new stopwatch
prove securecontextimpl equals works as we want it to in light of httpsessioncontextintegrationfilter s attempts to avoid unnecessary httpsession creation
include host when executing curl against unix socket
add missing declaration
add osx native fullscreen
exposing httpservletrequest response in servletserverhttprequest response
just show simple error message not cipher text in case of error
remove requestid from request events json
some still make use of the horizontal scrollbar from time to time
restored functionality for debugger inspect window
updated glm weights test to use explicit lambda
fix bug with master stopping trying to be master after he losses it once
add reset to pagingpredicate
just correcting wrong logger class
resize general config pane along with the outer window
remove debugging statements
add since to the getticketgrantingticket
cast the output of _train get to a frame
fix transport stops
updated the loop in shift binary method
remove useless tracking from searchactivity
add setwriter method
temporarily workaround rendering issues
separate can t run message
remove dead code
abandon long press on scroll clear hit on keydown
fix wrong import
add javadoc for view focusable attribute
send the partial result with oncaptureprogressed
updated log level
reverts removal of jsonignore on event type field
checkstyle fix add file header
allow suppress missingrequire to mean suppress strictmissingrequire
modify the member of fileinfo from public to private
remove thist from api
hotfix annotate stoppablefunction as publicevolving
fix read access assertion
remove a line of logging
reload only needed maven libraries on start
comment out references to deleted classes
add optmizations for cleartext to avoid sslcontext
remove private method from the diskstorage interface
change disable timeout for bt disable stress test
remove assertion ensuring pdf version of docs is present since it isn t anymore
fix quarry chasing water
make boundcolumn inner class static to reduce footprint
fix bad comparison c o findbugs
fix revolutejoint getlocalanchorb
romka merge override implement method by default
bring back watches toolbar on the left for now
disable markers in
add log to the simple account registration dialog
fix bug unknown database
replaced invalid entity nsbp with simple space in java docs
add missing mbean method
made tvals final
add override of the cancelpressed routine to cleanly close the add edit dialog
fix eclipse warnings
remove debugging printout from softcluster
remove another wait
fix broken tests due to uninitialized roots
add a gservices setting to set the rate at which the home page refreshes
pass exception cause on statestorageexception to simplify debug
modify observablecollapser jmh test to allow multiple response per argument
fix check for the max number of attributes
add close for mongodbclient
fix failing unit test
okhttp should use plaintext in transportbenchmark
hide secondary toolbar on startup
improved slightly the exception message to ease debugging
incorporated pr comment
add cleanup mechanism to launch plugins
fix some tck failures with the bio ajp connector when using a security manager
fix capsule javadoc
check npe condition
replaced character with string in sqltypemapping
remove todo and improved logging
do not leak project from javaexternaldocumentationtest
add possibility to remove the selection via the drawer result object
check cslm too
remove override stopufs in localtachyoncluster
adds property to hide call history button
remove selection from components hierarchy tree
sending future authorization
add debug info
add packetcollector getcollectedcount
fix broken test
remove unused imports
remove unused code in tasklistfragment
make json response parsing overridable by children
activate editor before hector popup
closing mailbox folder can cause messagingexception on some mail servers that block the folder
fix leak in demo app
make it possible for papplet subclasses to override exitactual
fix compilation problem
add null check for urlstring in gethttprequest
updating the test to gte more info when failed
moved oilpopulate to the worldgen package
extracted locale as a class attribute
avoid aioobe again with over large correction for nudge in corner cases
fix referencing am debug info
add tests to test suite
don t swallow the stack trace
enabling one more test
turn down the logging
add more debug info for assertreadaccessallowed
eliminated unused import
use all files from recent files in switcher if tabs placement none
documented enums because matthias cant seem to decipher that eq means equals
don t clear sem cache manually
fix bug in strokefont
add join as table keyword
remove remaining debug system out in spotlight
don t register rest actions on transport clients
correct the order of arguments
saving blacklist and hold for moderation lists when dismissing dialog
prevent possible backward steps in parsing numbersequence
add clone method for rendertarget class
do not switch language if the new one is same as the old one
remove unused import from bootstrapfortesting
fix a possible npe when a contact has no status icon
add method for updating import statements of a moved python psi named element
catching illegalargumentexception when unregistering receiver
don t output recommendations when there are none
improve logging wrt xmpptcpconnection shutdowndone
add dstest manual creation not letting derby create it
started support for the basic instant messaging
exposed two more callbacks for preview mode
fix wrong mode for fingerprint unlock when bouncer not showing
cache avg put time calculation
slightly improve debug message
hides empty view on refresh disables swipe to refresh for people list
fix a copy paste typo in viewpropertyanimator
add more descriptive backwards compatibility information
add spring boot ratpack groovy dsl example to javadoc
add support for customsmackconfiguration class
log an event whenever we lose the default network
remove unneeded check
changed icon for analyzer
add javadoc for cordovaresourceapi
reduce memory footprint by resurrecting the removal of childchannelexecutor
better wave underscore fix
don t change java identifiers when sanitizing
remove testoperationredo from problematic
fix wrong name
layout explicitly initialize default spacings
update comments in ufs block instream
handle null shapeids
remove unused code in blockmaster
set desired parameters in inodetree
rename domain to load balancing group in the log message
add logging of a disregarded exception
remove unused imports
gr should not hide java
sourceformatter remove unneeded code
fix bug for updating mcurrentposition
properly report number of markers in a statement
add license headers
fix ticking need to register the basemod tickhandler sigh
adds command line options ga_hadoop_ver and ga_opt_out
prefer direct buffer by default
check restricted for file name instead of method name
remove unused toplevelsearcher method from fetchsubphase hitcontext
log type of unexpected quorum packet in observer loop
fixes the lag when changing accounts in the conference invite dialog
add espass mime type
getting force close in incoming cal while recording the audio
disable hw acceleration for status bar
fix detection of decimal and long number in native number parser
fix ssl config error as a result of attribute refactoring
add config type serializers properly
do not reference testdescriptor
reverted the split up of the test setup
restore parameter annotation codegen
remove unnecessary null check on stdout stream
add packet tostring
remove incorrect import
mssql can t reference objects through the catalog name
add related entry framecount to saveframe reference
add a note on setperiodic vs settimer
fixes cut off audio conference window when creating a video bridge conference
build layer immediately when setting it in viewpropertyanimator
fix possible integer overflow in mtpstorage
make hostandport serializable
saving a contact with an email type of work sets it to home
store the collection name instead of the class name
customize tx tostring
fix compiler issue from last commit
fix another typo
clarify another javadoc
fix quality flaw
remove debug output and change log messages
add missing copyright header in foo
improve stability of progressloggertest
fixining what idea formatter did
fix compile issue exposed now that proper dependency is used
prevent deadlock during startup by add uniqueframeid to typemap
set default paths for fix static version
handle null correctly
minor tweak for windows testcase expecting illegalargumentexception
add default case
debug validation of processor ui panels
add pending drafts ignore list roll call when drafts are to be published
revert actfminvoker default url to android localhost
fix some quality flaws
defer heal when device goes back to sleep
add key_carrier_ims_gba_required_bool flag
if we re skipping an epsiode and keeping it don t mark it played
add acra version in field user agent
fix another npe when missing serialized waypoints and fix having different names for write read
expose isclustersafe and islocalmembersafe through mbeans
remove unneeded todo statement
fix possible npe when embedded
missing constructor required for system class loader
on alias resolution in pathresource ignore nosuchfileexception its a
android update modify devtool
don t show firefox cmd click warning on mac seems to work fine there
increased timeout time for cancelling tasks
fix cache test in osgi
revert don t re launch activities while the device is sleeping
learning curve now has configurable metric
call appopsservice from the handler thread
correct sized icon for plugin module type
add missing runwith
add drop and ability to specify what kind of index migration
avoid npe in debug logging when deleted cf is dirty
fix javaclassreflitest hopefully
create copy of typeinfo set in dataflowinfo or
add javadoc for class
test compare strings and strings leftover from path api refactoring
add packagedevelopment context to list of commands whose keyboard shortcuts are processed even if focus is in terminal
remove the test for the wrong encoding
add comment describing class
do not raise a conflict for a synthetic methods e g enum values
remove final from signature
speed up dissoc on array maps
add override to hush warnings
remove some debugging code
switching to context classloader
add annotation for root relative metadata
add description for barbarism category
ensure acerender icon destruction scoped to that editor
remove usage of guava method remove in future versions
only keep reference to selected message if necessary
fix the test since client s heartbeat interval must be configured to check heartbeat
deprecate methods that are used only from one rule and thus probably don t need to be public
remove content type
popup menu called from keyboard hides navbar
don t use this as the lock in reader and writer
allow setting the drawertoggle without toolbar
put bindposetangent vertexbuffer type enum value at the end to not screw with ordinal values
fix stackoverflowerror in internalloggerfactory getinstance
rename method for clarity
add support for resume_on_broadcast property with atmosphere cpr
fix settings command help msg
add an executor service that automatically retries tasks on failure
fix whitespace in virtualmachinemetrics
revert previous patch
improve foreignkey tostring format for diff
catching throwable instead of classnotfoundexception because class forname can also throw linkageerror and exceptionininitializererror
tweak theme preview to eliminate horizontal scrollbar for default font size
contact list accounts avatars hidden correctly due to settings
fix bug in let generating multiple this locals
fix regression that would not show clear devices
fix col doc bug
remove accidental main method
vfs refresh bug
mods expect a null saves root directory and account for it
save one common string allocation
make sure filebasedindex will re build if building vfs from the scratch
add type information to variableexpression
make inlinegetters option package private
implement shadowfragment getarguments and setarguments
add info on how to drop default interceptors
renamed audionotifierimpl to audionotifierserviceimpl
add some todo comments
remove head line which was left after fix merge conflicts
throw an error if a switch data pseudo instruction is encountered when auto fix a method
fix networkmonitor logging to include netid
remove detection for slf4j noplogger
readded deprecated methods to keep api
don t include email followers in getfollowers query
add fudge factor to initial spacer height
hook up webview clearsslpreferences for the chromium http stack
fix a bug when detecting cookies name with sign at the beginning
find method add
not share fixedrecvbytebufallocator handleimpl
updated assert equals to reflect changes to view resolution
add notnull annotations
git remove obsolete javadoc
only update view if force layout is set to true
fix typo in supercolumn mostrecentlivechangeat
fix potential npe
handle non started layouttransition animations correctly
refactored lookup of annotation method to use name instead of type
fix format of logging of exceeding bytes
fix multimap migration rollback
making delombok compatible with post resolution transformers meant delombok would fail with a duplicate context value error
add a timeout for the udpclient so that the test cannot hang
skip null service names support info
detect executed and skipped task in debug output
remove no longer required function abbreviate
convert println to logging
do not delete remote messages during sync
remove unneeded constructor
add method for finding the first in an iterable matching a spec
fix raptor index insertion
annotate http2serverdowngrader with unstableapi
core only add view to parent after measure and layout
fix a broken import in multivaluemode
enable sync tag editing by default
fix sample code in javadoc
upload correct txpackets count
add outline support to insetdrawable
increased timeout value in hystrixobservablecollapsertest to accomodate slow travis
remove excessive sync icon
fix a potential npe
touchdown is now caching the handled state of the event before the
revert fix for deprecated method to avoid crash for older recyclerview version
fix xsd test to include wildfly schemas
remove some unnecassry methods from invocation
add assert that repair keys arrive in order
remove unnecessary public modifier
don t print extra space before finally
process only those kotlin files that sit under source test roots
remove activiti endpoint from the default proxified endpoints
get tests working with ant as well as ides
cachenotexistsexception contains name of cache
switch the drawer content back to the normal draweritems before fireing the event
add some comparisons
avoid npe in autowiredannotationbeanpostprocessor
remove unused field from alertdialogwrapper
using concurrenthashmap to prevent concur mod exceptions for cache of simple names
fix country code for netherlands antilles
fix potential deadlock in cdc state tracking
catch runtimeexception explicitly so that it doesn t need to be wrapped
change zap api to return different status code on errors
expose content uri
include root comment as qualifier to get root fqname if failed with short name
use the right version of classutils
completed documentation of the extended listener methods
reduce jfreechartdialog maximum zoom
add new createtimer action for scripts and rules changes due to teichstas review
fix parsing of multibyte chars
remove unused field
add option for configure if heap memeory limit is committed or max
changed loglevel of success notification to debug
fix npe related to people extras in kitkat builder
tree update bug
generate tests for ie
allow top level drawable container in progressdrawable
not getting gl context on reshape
fix license header
remove irrelevant comment from dynamicfbosmanager
changed wierd class for standard library one in memoryprofilingrunner
add check of right class before removal of edges and vertices
correctly handle custom folding desc attribute followed by defaultstate
remove unused import in scheduler java
remove unused imports in androidnet
fix bug in foreignkeycolumndefinition
fix merge error from master by setting new method on servletholder to source embedded
don t launch broadcast receiver processes during boot
eclipse requires some tweaking
add package version code to mixpanel properties
add tostring to bsonbinary
when displaying the actual class name use the simple name
fix typo of previous patch
tweak nomenclature for special debug opcodes to match dex format
fix unit test so it runs on all platforms
add final to ep_name
try longer waits to improve stability of integration tests on travis
removes debugging statement that accidentally got checked in
expose cookie api
fix a search bug for wrapped window callbacks
don t declare them as nullable
add comments on code
log the default charset once during charsets class initialization
fix failing test
do not allow suspendintermediarydeallocations after free
fix udc memory collection on ibm jvms
made messagesettings class package private
fix case on shutdown when the pool is null
add failing test
extract the xml from the overall result
add a better exception when hist encounters a constant numeric column
handle empty bigtexts more gracefully
add a useful debugging statement
add tofullstring in version to help troubleshoot
remember dependency scope for any library
set multiplier to two
add missing stencil buffer rewind
fix java docs
add test to check jedispool with password
add missing copyrights
add serialversiouuid s
add extra tests check for blank strings
fix mssql case sensitivity issue on sys indexes
createframe get post
add a test that would have caught the gson breakage
remove timeout against remote connections
made transformation class non final methods still final
initialize mshowwhen to true
fix minor bug in binning clusterer
remove user property since lifecycleparticipant does not read system properties add see
java use platform specific new lines in output formatting
fix event dispatcher timestamp sorting bug
fastscroller now reports scroll state changes for its associated abslistview
use renamed image scale type
remove unused imports
fix simpleblendmaterialsnode s desiredstatechange
fix menu item position issues
translog remove useless random translog directory selection
enabling users to instanciate a sharedpreferencesprovider without using guice
add hiccup testing module
remove unused method
add translation lib version to debug info
add back putaaa4 int
fix scheme to use prefix
remove empty return annotation
simplified prefs storage
improve mathjax sizing behavior on windows
make compactstringobjectmap serializable
fix incorrect resource in jid
replace addlistener with removelistener inside toolwindowmanagerimpl removetoolwindowmanagerlistener
fix comment typos
add support for setting the user agent
tidying up javadoc
git don t report ignored packed refs line even in debug
turn off walk limit when no transit mode is selected
updated settings comments
add convenience for get requests with no configuration
bump version for a rb
remove deprecated and unused getbeanclassname
defaultmessagelistenercontainer immediately invokes stop callback when not running
add principal to subject
fix problem with https in tags
add argument to exception constructor invocation
don t use instance scoped variables in try with block since it will get
fix weird qs edit bug
also using nano time for dl seed just in case we launch two models in the same milisecond
missing semicolon oops
set timeouts for accessing wikipedia api to hopefully make community languagetool org more stable
fix potential infinite loop in abstractframedchannel
take a shot at the screen rotation correction
throw real outofmemoryerror
use correct log path when deleting old thread dumps
fix null in shared link
resizing tabpane with the jframe
don t grab focus to ant messages upon build completion
formatted my code to follow code styling
do not leak locale alert dialog
remove obsolete pop operation
add class loader exclusions for guava and apache commons
add license header to filedownloadflowthreadpool
rename getscenefinalrawdata to getcolorbufferrawdata
skip a new isolated world test
remove unneeded cast
add missing super teardown call
improved sqlserver definition based on forum post
remove support for attribute map parameters in callback functions
keep track of the state of the fragment and do not show deletesitedialogfragment if the target fragment was paused
fix butcher text in the case of only one mob being killed
remove unused imports
fix wrong shard spec
registered shader node definition locator to the android asset manager so that shader nodes work on android
missing some tests
prevent nullpointerexception when no marshaler can be found for a parameter
restored original android proguard outputdirectory
stringutil capitalize fix
remove pluginstate installed state
remove unnecessary synchronize on initparser
add support for cdma conference calling
remove additional rdd cache
fix focus_backward search if view had no id
remove old code
add createcacheconfig method into the cacheclusterwideiteratortest to be overriden in ee for creating native memory backed hd jcache
fix up directory injection of log file locations
make shiftingmagic use animate parameter
updated param name to match type
entering branch name does not finish gitswitchbranchesdialog now
java remove unused import
java generated formatting
don t set notification mode to background when on pause
don t show clean actions if there are pending snapshot restores
splitting annotator and inspections
fix copyright header
remove unused log message
explain the void send throws ioexception
delete the temp commented functions
minor add empty line
prohibit ticket updates for empty repositories
rotation lock bugfix
also compare drawer items with same counter value case insensitive
remove unnecessary code
turned off gradient based auto lambda selection for now
support generalized light element conversion
remove unneeded constant
fix a possible bug in textureatlas java
set initial capacity to avoid array copy operations
kill pommodel from projectimpl
add more logging
make text field lower
use crashlytics to report oom errors
fix broken unit test
improved error msg for unresolved classname staticfield
splitter doesn t honor minimal size
remove an unused method
refreshing output roots progress message
fix compatibility with new server
improve test coverage
remove old lombok stuff to bikeparkunlinked
remove settings_system settings_secure and settings_global from the default report fields due to privacy concerns
deprecate unused constants
fix touchslop changes
fix nullpointerexception when testing equality of httpmessage
fix nullpointerexception in drawablecontainer mutate
remove an override that is not needed
fix exception based logging to actually log the exception
add missing model version bump
changed mixedbulkwriteoperation to take a list
do not hold a lock when calling api to grant default permissions
remove unused import
remove an unnecessary sanity check
ui index generator should explicitly exit vm upon termination
do not store password in memory provider if save password checkbox disabled
specify sandbox attribute for html preview iframe
remove unused method
simplified list of login methods in xmppconnection
documented incremental java compilation in the dsl for javacompile task
extensions use couple
debug appearanceui events
use writable map config
add more zql erb template test case with conditional branch
allow atmosphereresourcefactory to be defined in meta service
implement open file
remove dead import
fix swing graphic bug
ninja remove bytecode printing
enable bytecodeanalysisindex by default
fix wrong index in colorspacetransform
add license statement
fix import statement
adjust error message in abstractaccessibilityrecordassert
add asynchttpresponse getrequest
addeding try catch to card device parsing in alsadevicesparser java
when printing cache full warning print the sql statements in cache
standardize all newline chars
replace stringbuffer with concatenation
keep lastconnectionfailure till getandset
provide a faster way to get a location
remove unused imports
fix a typo in a the comment about the auto argument
remove obsolete workaround
reintroduced erasure check for method graph compiler to make it more robust for types with illegal hierarchies
fix nullpointer in localproperties
only use hw layers for fragments on jb
properly set transit flag for caching raptorworkerdata
add iscanceled check before sending response
deprecate a method that shouldn t be public and once that s not used
remove jspfile usage
remove dead code
fix bug in instanceprofilertest
log sdk for missing methods
should not need batch begin and batch end in resettransform either
add getreachedvertices method to streetrouter
catch all exceptions thrown by xml pull parser
set a threadfactory to dhns event bus
perform an action in accessibilitymanagerserivce using wrong process id
propagate project from com intellij ide startup filecontent to com intellij util indexing filecontent
incomplete assignment stmt does not produce a soe
add comment header
addressing a review comment
fix logger class
simplify the echo client example
fix a server side crash
replaced deprecated string constructor
add an idn equals method to check names
tools that print the root cause will provide bad info
fix data type in setstreamversion
add typing to context handler mbean and remove import from xmlconfiguration
remove unused code
fix plugin command name in remove plugin command
checking read access allowed on gethighlights not event dispatch thread
set copyright to google inc
fix the documentation
fix issue with fixedlengthstreamsourceconduit transferto
fix the lexicalizedparseritest
add missing builder endobject calls
fix paramter order expected actual
fix for polymorphic cluster security
add more useful output
log exception that is ignored because of execution in thread
add usage output when parameters are missing
remove debug output from mixiniteminworldmanager
delete ssh public keys when user is deleted
add the hint critical to task executed by backendimpl after big import
fix compile issue
make autocorrection work in suggestionautocompletetext
add pkpass mime type
add a new indexrepairjob counter
add missing np check
introduce filed constant from null
pass the original exception as the cause when throwing connectionclosedexception
remove now untrue warning on the parser
moving alladjacentcombination to recursion package
allow internal plugins to register parametric types
fix class name serialization bug in serializablenativevalue
patch to make test cases appear more like agiledox when running them in a report or ide
createsnapshotfor accepts enginewrapper
improve doc of api ce queue
remove duplicated and method in and response handler
add copyright header
improved error message for bad arity
remove some performance test result schema migration stuff
fix session to clear cookies on logon
add selenium ignore
improve console logging on startup exception
deprecate unused code
handle more gracefully the case where a texture
move ntp updates outside locks
enlarge type information in ref graph
deleting commented code in hbasestoremanager
handle normalization of a null phone number
if an image is in the cache don t fade it in
add more error message text to help with debugging
do not index file content that fileincludeprovider does not accept
deflake metrics concurrency test
allow the projectfilesystem to be passed to the testsourcepath
use show this page from default settings
add debug code for hudson
remove empty check in doxcontent parser or validate will take care of this
project thumbnail should have the same aspect ratio
restore a currently unneeded change
simplified dsl interface hierarchy compatibly
don t cache closure type in groovypsimanager because its return type is already cached there
fix typo that makes apps on sd work better with app widgets
add instance config change listener on helixbrokerstater
fix a test not marked test
remove unnecessary cast expression
pause the underlying socket
synchronize chats map
add build dependency on binary spec instead of build task of the binary spec
remove negating comparator in topnnumericresultbuilder
remove unused code
rename to the correct method name
improved exception message when a class cannot be created because tx is active
remove stray import of java util list
fix head cancontain
fix configuration test
fix typo in eurekainstanceconfig
fix stupid npe
check that maven module already has library in dependency
remove neuralnet tests
relaxing timings to accomodate travis after she has had a stressful couple of minute
destroy display lists when running complete memory trims
rename max_in_process_pages to in_process_reschedule_batch_size
renames callparticipant to callpeer so that it would better reflect our new call architecture that also includes conferencing and conferencemembers
set stamping to false by default i e nostamp
turn off r diagnostics by default
add common method for windows new look detection
make blockmeta size really measured from actual file length
indicate immediately when planning widget is in edit mode
add missing types to managementtypeutils
fix a bug where list wouldn t refresh after deleting from edit page
made brick brick block not brick bar
add optional prefix to bike park names
make tuples final for better performance
remove unused code
reset in dictionary stream when opening the stripe or row group
fix issue with updating props quickly causing released camera object to be accessed
disable fs cache
add missing final to moco exception
incorrectly pushed the logging change to master
fix a default clusterlisteners in simpletcpcluster
only moved comments
no more document commit thread
moved double check inside null safe test
fix arrayindexoutofboundsexception occurring under certain conditions
remove tostring to
remove unused imports and parameters
fix javadoc error
need blank ctor
deprecate socksmessage encodeasbytebuf
add testcase for in visual block mode
fix moved update calls
fix changed signature for physicsrigidbody methods
fix srtm free plugin
fix copying char literal to clipboard
replace see tag with since
fix inaccurate typedarray getdrawable documentation
check style fix for gradleconnectionparticipant
remove unused var
renamed the test case to better fit into its intention
fix mockito use of renamed ios method
fix bug with not picking up the latest maven snapshots
change the runtime exception to a log info if interface doesn t support multicast
throw mapperparserexception if trying to parse value as object
fix precondition error message to use s instead of d which is not supported by the preconditions formatter
remove deprecated code
improve error message for cassandra stress schema errors
don t throw a parameterexception for a regular exception
suppressed leftcurlycheck s deprecated field
fix mixed up episode and show text view in upcoming
add new fixes to upgrade message
fix checkstyle error
undo accidental checkin
remove decoding flag
fix elevation handling for cases of truly one way streets
don t run dsym on bundles
make crash tests shorter in default setting
add mimetype property
ignoring broken test until jonathon can comment on the correct logic
follow coding style
test new routing provider alpha routing should work with java for quite long distance
simplify method slightly
allow compilation of classes in appcompiler to be canceled using thread interrupt
fix lint warnings in destructorgenerator
fix incognito mode
slightly improve iszoomingallowed
make everything conformant to the changes in the localvcs
remove unused imports
settings file templates doesn t follow theme colors
remove netty import
remove redundant check for boolean
allow messageaggregator to disallow non empty content
required exception class
make sure the connection always closes no matter what
fix missing imports
socket closed errors
enable symbol completion failure stack traces in tests
add a note to the doc of webview loaddatawithbaseurl
make checkstyle happier
remove obsolete statement
make unmanagedthreadexception an executionexception
add a little support for two pass rendering
remove unused import fix checkstyle
throw correct exception
change popupmenushowalerts to cleanup resources when no longer needed
fix issue with double spaces in command handling
do not release unacquired semaphore
allow plugins to customize the version label
initial hibernate type support
removes incorrect comment
hooked up radio buttons on tablet view to switch fragments
trigger another validation build just add a space
diagnostics for focus timeout saves allocation in focus command
reduce lock contention remove unnecessary synchronization
fix a bug that caused an npe exception which occured when ipo had no curves attached
fix possible capacity exceeding of file buffers
remove add activitymonitor during activityutils finalisation
fix perf tests after recent changes
hook overridemessageproxylistener into the proxy
add option for ne
add the last catch to add the group bys in the list at the end of the loop
fix that standardsession getlastaccessedtime uses correct exception message
turned validatingedittext back into a custom view
avoid boxing of the number of bytes flushed
handle null results in basicshortestpathtree getstates by returning an empty list rather than a list containing one null element
avoid concurrentmodificationexception in singleconnectionfactory s aggregatedexceptionlistener
set content type on error page
not reinitializing the backing arrays for matrices and lights
fixes display issues with the about background image
fix a bug where task rabbit would be hidden when ideas loading wasn t enabled
implemented filter that copies across resources so that views resources etc can at least be reloaded
showing welcomescreen after project loading cancelled max
catch any exception and properly display the uri which caused the exception
skipping some checks for abstract value type if modifiable but not
log internal server errors in localmucuser
add defensive checks for bad setfps api behavior
fix issue that can hide slow network requests
add more informative usage with examples to verifysegmentstate
fix very theoretical timing issue
remove unused field
corrected zapnumberspinner arguments
delete a fairly useless unit test that was also non deterministic
add the ability to disable opengl drivers preloading
extending async testunit timeout
fix html tags in comments
remove some debugg syso
changed the message
adjust homebrew go sdk path
remove unused declaration
fix negative padding for colspan cells
fix title visable bug
this makes the appropriate step call now the problem exists in cucumber junit
intercept the otiose event enqueue
catch cancelled exception when after change lists loaded correctly
remove unneeded checks and add javadoc
do not perform magic with classes fqns
make blocktank extend blockbuildcraft
fix itemfluidcontainer always draining the maximum amount regardless of the amount remaining
fix missing recids in txmaker
handle javac fail by oom
add more example dirs to examples win for js mode
streaming api breaking changed returntype of windoweddatastream s mapwindow
unit tests support windows
fix javadoc link in connectionconfiguration builder
update testblenderloader to test loading relative images
disable conditional code warning for consistentidgenerator functions
add getresultsstore hook
remove unnecessary warnings
changed error message
fix null point exception when on back pressed in file view tab
fix typo in compositeexception documentation
fix copy paste error and throw correct exception
fix return type
fix quality flaw
clarify entityscan deprecated javadoc
save password workaround
reverted change in textureregionlibrary
add some more verbosity and a comment on what would be a useful flag
fix a bug with using arrayadapter addall
add legacy temperature sensor type
fix a comment
fix calculation of relative path for regular files specified as an task input or output root
fix zookeeper exception logging
count maxrowcount in if applicable
replaced json parsing error with a simple one
remove the system out from my last commit
add flashfox lightning browsers to white list
make tenantjson extend jsonbase
do not drop unknown run configurations with same name regression fix
discovery fix membershipaction sendjoinrequestblocking should now expect an empty response
add additional data parsed from tmx file
throwing proper exception when no extent letter is put to font
do not write empty line to existing changelog when no changes were found
improve robustness of task manager test
mute failing assertions in indexwithshadowreplicasit until fix
update xpath icon size
add a few more file extension type identifier map entries to pbxfilereference
clarify capture order for reprocess
quick definition popup now remembers its size between invokations
add test for merging socket servers without port for both server
add fused location provider to real provider list
bring back operation rollback on unexpected mapping change during recovery
add tooltip with full path to project toolbar button
readded import statements
if we get unsupportedencodingexception include it as cause in the configexception
fixes a misuse of netty where a fix size thread pool was given to a
use goto nonprojectscopedisabler ep to show hide non project items checkbox
make incomingtcpconnection responsible for version handling
use the original session by default when cloning a request
revert this change for now
print only non static internal members
add pagesink to spi
set namespace aware parser for test results
fix npe on input launch on server
add tests to prove that properties with dashes in the names can be found
fix some typos in comments
add some additional class comments
escaped instance of symbol within a url so that more apps detect the full url
throw exception if composite subfares list empty
add getcontexholderstrategy method to securitycontextholder
add url target to allow add events with a get call
fix other kafka test
move the selection handles vertically to match textview
selecting hive on hbase table may cause filenotfoundexception niklaus xiao via ashutosh chauhan
add spring context support jar to dependencies
fix shift precedence in fontlistparser
fix import order
improve packet javadoc
changed headfinder for whpp
remove help toolbar icon for now
fix npe in verifier
fix initial status of jabber provider
changed log tag to eventbus
allow elastic averaging parameter to change during checkpoint restarts
stop all sounds when a call is ended
don t log metrics for initial screenshot
fix default favorite color
integration tests code cleanup
add test for having without group by
moved bptblockutils java
remove redundant hashset iterator allocation from the singleresultset
don t execute thread if we don t have any notifications in the map
fix a bug in android keystore updateaad
fix issue with entering end tag for html xhtml jspx files
remove unnecessary suppresswarnings unchecked and add it where necessary
prevent multiple timedtransferedges
cleaned up a comment in the bst
add license header
add a comment about possible exception that should be monitored
fix typos in json documentation
update comments from sam s version
delete a dead method from walletprotobufserializer
fix asymmetry between phone filter and email filter apis
remove unnecessary parent loader as argument of dispose
repeated ctrl space should focus lookup even if it just fills common prefix and even if it actually doesn t
fix potential problem with immutable points list in line model
fix bug double displaying if image is cached in memory
enlarge pingpong test timeout
do not remove contacts not returned by the server as they might come later
fix settings caching for ui
move unwrapped return type specialization from abstractconfigobject to configobject
making the object less chatty
drop unused return value of getepisodesandupdatedatabase
put a note in about not caring about blocking io when managing the ratpack groovy script
simple gone fullscreen button
fix white bar in mac new project wizard
do not create empty env variables
selection state now affects the target
add a comment
ignore special duration notation
clone the drawable while setting it to a subactionbuilder in case it might be used by multiple buttons
remove unneeded and potentially incorrect check for mac
remove broken imports from simpleentitycontentprovider
make deeplinkerror methods final
voicemail number setting is throwing exception
add cap_block_suspend to system server
throw configurationexception instead of just logging a warning
fix up tostring of mongocredentials so it doesn t print the address of the password char array
fix classifier lookup
add command logging in for non zero exit code
fix windows test failures
take screenshot of finishing apps as requested
fix assert in iced hash map
fix checkstyle issues
remove back branch preemption
add convenience constructor
create default package for library module
fix unit test assertion
fix calledfromwrongthreadexception in mediaplayer s subtitle support
fix a log statement
adjust pattern in structureddocstring to work with restructured class definitions in docstring
notify adapter for new items
add hashcode method
smallfix add output to buffer class name
fix eager return causing the lock to not be released
allow endpoint jmx export to be switched off
fix an infinite loop on a certain nio impl
compare creation times of records if access hits are same for lfu based eviction
remove killprocessaction from debugger s toolbar as it now duplicates functionality of the stop button
fix null handling in boxarg
expose to the platform the ability to pick up changes in child visibility
fix avatars binding in contacts
fix inconsistency in clock calculations
fix conditional bean defn with yubi webflow
fix project encoding in convert to basic latin intention test
cancel drag sort when adapter is replaced
delete an unintentionally copied comment
remove whitespace from activesession
fix potential npe
fix directon drawable for poi
updated resolvepath javadoc to reflect environment based placeholder resolution
change the name
remove superfluous suppresswarnings unused from putalloperation and putallbackupoperation
remove some debug code
add check for empty attributes list as well as null in delegatingmethodsecuritymetadatasource
fix checkstyle error
remove an unnecessary animation
deprecate auto indexing api
add since to flushstatements and remove uneeded public
make private inner classes static
add extraction of else branch of when expression
fix splashscreen memory leak
add some binary expression parsing
fix default image not displaying prior to load
selecting text from a punctuation sign selects previous word
don t manipulate selection before find in selection
don t use stubbasedpsielementbase getstuborpsiparent
stop complaining about failure to remove routes
forcing termination of pool when timeout is reached in concurrency test
malformed elasticsearch yml causes unresponsive hang
handle special case when called only with one lambda lambda max just output the null model
rename method to clarify which connector is being created
check our assumptions about the cache before making assertions
find subdirectories of libraries by system independent paths
remove unnecessary activity finish when dialog is dismissed that was causing a duplicate finish error to be logged
don t paint mac bg with darcula switched on
fix cursor leak
remove useless cast in array tabulate
fix state loss
test use a pooling connection manager for rest tests
do not apply single change multiple times
remove already used named arguments
set confidencecalcmethod for nontransit searches
add a warning based on what appears to have been the fix for an apr crash after the recent refactoring
fix npe when stopping the server while users are connected
remove jalopy import line
fix photo reshoot
only throw nosuchalgorithmexception for getkey
add standard atmosphere license header
properly encode multiple headers with the same name
providing correct folder for configuration
actually allow all valid characters in a wi fi ssid
check all files for defining availability of gocheck framework
made the bean serializable
tweak access modifiers
create uris directly from the file objects
fix connector shutdown
use debug logging if no translog file is found
add new getendpoints method
fix incorrect lens position
make connections setreadonly method compliant michal wegrzyn via sergey shelukhin
throw a more meaningful exception for a missing font
set self type
moved logger field up
add a constructor that allows wallet less operation now possible due to the previous patch to support multiple wallets
temporarily rollback previous change pending further investigation
allow use of hibernate order object
defer building authinfo to after a refresh token is obtained
remove false check on installation
fix missing param doc
add wpstats tracking for stats
add lt version and build date to output as a comment
fix wrong logic
remove filters when idanmakuview is released
fix potential npe in telecommanager placecall
add video link
protocol will call recycle
cancel job if we re shutting down
apply patch small improvement
add where to tableorview interface and updated testcase
dispose tree builder in disposeuiresources to prevent memory leak
observed strange lock up on osx when two threads were writing concurrently to this map
fix kotlin maven plugin compilation
don t kill call on transaction timeout
implement a better tostring for patternrule that includes the sub id of the rule
add xml reference for format and tostring
remove unneeded rebuild was toying with where to put it
don t create launcher script
remove unnecessary code
create emulator files under project root not in tmp directory
displaying the bone name in bonecontext tostring method even if the bone is not yet created
returning editor to previous state after test
temporarily disable a couple of non deterministic tests
fix embarassing typo in variablecontext with
changing the greeting part from decorator to be bold
do not return lowercase role names when mapping id or role to its name
let prettyprinter hook into renderend rather than executestart
if user specified set transaction isolation level
handle date conversion better
create as method uml action
set account setting crypto app to apg by default
add notes to setprotected javadoc
fix the overlay effect problem on android l
fix trivial typo
update syntax to use hashmap
fix errormanager ignoring error code in messages
fix virtualmachinemetrics instance
corrects the bug where the primary shortcut is overwritten by the secondary one
fix wrench to pass sneak clicks to blocks
fix javadoc reference to maven repo local system property
remove background in episode details activity
use a bit more descriptive name for a parameter
swallow exception caused by dismissing action mode
remove diagnostics sorry
add proper deviance w y f for glmmodel wasn t overriding base class implementation with gaussian
remove show hide unversioned files action from popup in commit dialog
add carrier config variable to hide preferred network type
add path methods in treenode
fix previous test case
reverted abi change in abstractcamelrunner
fix typo for setting selection drawable
drop unused imports
add a unit test for https to https redirect using relative location url
show no network available message in plansactivity
fix javadoc warnings
do not select first raw when checkbox selected
remove uncalled function java counterpart of
remove unecessary method
enable cache monitor on computes by default
renamed inner class
add deprecated to webmergedcontextconfiguration constructor
prevent test failures under windows
allow non ligthvirtualfile non physical files to be considered in project
towards remove lombok
fix background color in maven tool window
testing httpclient fds
call homeisup in the android actionbar example and use a string resource rather than a hardcoded version
remove import in indicesclusterstateservice java
make qualifier mutable
remove errant system exit from stress so it can reuse keyspaces
change back order of import statements
creating highlight info anyway
don t swallow assertionerror from changeprovider
prepare for the message mediatype
copy visibility docs from support version to base notification
add test for gettype against an abstract factorybean
make the cached compile scripts a soft map
disable intention for unresolved decorator
launch the settings activity when the settings menu item is selected
show the notification with a delay to avoid blinking when too much output ceases quickly
specify the sampel support websocket
plugin does not save book encoding
add comment about why there are so many throws interruptedexception declarations
resetting formatting scope to develop
enable long click on recipient select view
fix the build
do not wrap ssl exception as eofexception
make loginterceptor public
improve some comments
fix doc typo
improved logging client readhandler i case of not beating
temporarily disable socket tagging
expose disambiguateuntil for subclases so we can roll up disambiguations
forced db in threadlocal in parallel mode
add missing space
fix bug in do at top level
remove commented block
remove unneeded refresh during post recovery
lwjglframe remove setvisible from constructor
add onkeysloaded callback
restore correct line directive file name
add systemapi annotations to action_query_package_restart and related
fix junit annotation
add property to force use of ldapcontext instead of dircontext
fix equals bug
fixes duplicate labels in chat room config form
generate empty name string for default package
revert escaspe default to true in querystring
log fix npe
add keys values to immutableopenmap
handle null molduser
contained i branches shows always the loading status
tweak recents launch app animation
changed member declaration
add clear method to transcript
delete row fix
extended web xml highlighting test
postponed profile creation after time
remove an empty line
corrected minor spelling mistake textursloadedcount texturesloadedcount
move kafkaclientprovider to servicescommon
changed exception message
stop preloaded build process on groovy compiler setting change
fix deselction of all gpx
allow greater flexibility for template method createdatabinderfactory
fix merges from regenerated r java file
remove unused radian property from instructions roundaboutinstruction would overwrite this field
make printer constructor public
add ok button title in the accessibility mode
missed custom json printer
add option to enable the lint checks in the debugger
add missing not
remove deprecated method in camelhttpclient
fix serversystemtest response content must not be called twice
revisit file too large policies in source tab
fix issue with forward merge and api change
fix route recalculate
also catch smackexception in socks5bytestreamrequest
fix off by one error when filtering application uids
nti fix an inexistent property spurious warning
change back tests
don t render template twice for append processor
fix problem with linkeset modified right after query in remote configuration
fix up channelpipeline javadocs by add lost don t
change peergroup addpeerfilterprovider to return a listenablefuture bloomfilter
add deprecated legacy method to serviceregistration
modify onedaydecorator to be cooler
consume the iterator to avoid endless loop
fix fatsequpdate invalidation
remove an unnecessary offline suffix in the helix instance data manager after a rebase
preserve tree scrolling position on last class level watch deletion
add getters for clustermanger algorithm renderer
don t add hostconfig as a listener twice
alter the history_projection and bookmarkcolumns
replace setopaque false with setcontentareafilled false
implement unbufferedtokenstream get int within marked intervals
help displays the wrong help page on startup
expand data chunk for satellite chunks with only data output
add missing package declaration
fix compole error and the doc snippet
ignore empty locale list when diffing configuration objects
fix failing tests
make exception message more descriptive when methods do not have a void return value
add compass to poi context menu
minor ignore android dependencies when looking for author
remove unused default content type
remove extra jars from make process classpath for non gradle project
caching tlds with uries
remove duplicate in metricexportproperties
remove unused import
darcula eats almost all edt cpu time
remove unused fields
remove unused getter
fix messages history loading
https download urls did not work correctly
increase the speed the badgeview animates on screen
read whole log from all branches
fix gpx utilties to log exception
remove unused import in defaultdatabase
use the same stop
add primary keys to tpch in testqueries
don t refresh vcs file status when entering exiting dumb mode
fix erroneous cast from byte to int in ibbinputstream s read
add ability to download model via rest
provide a slightly longer wait for thread termination
only ignore linkageerror on loading libc in filepermissionshandlerfactory
fix missing plural in a warning message when checking rules
add props to droid fu guys
do not increment access counter
remove debug counter
fix minoffet with scroll flag
renames output schema name from m to json
do not show warning about debuggable attribute
add file format version attribute to export file
smallfix remove unnecessary return statement in lineagefileoutstream
update the javadoc for the clip path fix
do not create lookup items for unnamed elements
add lastreturned method to resettableiterator interface
sends heartbeat interval in ms
fix merged comment
explictly set account status to offline when waiting for push
attribute value index check
fix exception in preprocessor
fix wrong null check of hot restart state
fix inversed boolean
use a wakelock for updating service on network change
fix description of leftshift task in error message
avoid deadlock when fstrim requested at boot
do not trigger a restart when a test class is changed
add some todos about possible future enhancements
add statement type as uri param
swap contact us and application log buttons
restore copyright message
add centraltoplimit property for hoppers
fix throw new runtime when error
animate photo to right place
remove all quoting of the tmpdir argument for the test harness
remove some fb internal task numbers since they are meaningless in open source
fix deadlock on shutdown
don t shadow lrucache since the full java source is available and usable
prove placeholders work correctly
remove dead code
add reference count number of times an identifier is used
fix altitude correction
add ide config path to root dirs
adds a getter for the gen jar
restore public access for class used in kotlin
fix sctpmessage duplicate behavior
fa add another potentially interesting rule commented out
fix custom font rendering
suppress unwanted commented lines
renaming local variable to camelcase
temporarily ignore this again
add retweet operation statistic
fix docstring file name
rewrote the channelhandlercontext explanation
loosen natlog relations for ccomp
add stringsdict file type for xcode project generation
don t merge downloadable messages
fix glitch in the innerprotocol specified for eap_ttls_md5
add test of appending params with null values
packettransceiverimpl remove isalive check
remove some unused code
fix npe by guarding the reference of variable
columnvalue does not handle null_type yongzhi chen via szehon
fix mobileplurkapi naming
fix reformatting on paste for short lines
fix change the way it saves a trends search
rename aggregatelifecycle to containerlifecycle in javadoc
stabilize remoteejbclientstatefulbeanfailovertestcase on windows
only set accessibility once per message not per delivery
make some logs not emit
add constructor with activity
deselect previously selected radio button in detaillistpreference
removes unnecessary setup in concurrentinstancestartupit
update code editing
reduce num of haslocation lookups in initlocation
add container null check
make job key consistent with model key
fix in modelbuilder offset handling
add mmediauri to the parcel
remove addendpointconfig method and it has been add to abstractserverconfig
state merging interruptibility
remove piece of commented out code
remove unnecessary condition special casing iostypebinding in statementgenerator
fix precision issue in gpuimagelookupfilter
moved http post processor bean name to beanids class
move test to proper place
fix java lang illegalstateexception when using as a fragment
improved error messages so a user can understand them
ignore illegal state exception support info
fix quality flaw es debug output
add getlongextra to shadowintent
also invoke validateschemaagreement for creates
fix doctableinfo test
fix a comment
supported labels for vertex queries too
add quotes around name
fix merge issue
add call to pubsuboptions close to close any opened output files
hotfix runtime fix processing unmatched build side when all buckets spilled
fix compile error
fix a bug in setidproperty
always updateinitialset allow multiple changes to initial set up to the point that the initialized event is sent
fix junittestrunner native code reference deleted unused variable
add explicit class exclusion to the routescollector condition
add missing return statement in pyutil deleteelementsafely
reduced forge requirement
fix mvn site warnings for tachyon worker block io
shortened the jar name
remove a duplicate test
fix unsigned infinity test in the testputrpc class as the latest version of
increase max parallel replica sync limit in partition lost listener stress tests
fix stbtt build shared lib name
fix compile error
don t load plugins from network when running test
fix pcg rng constructor to make it deterministic for a given seed
fix race condition in wakelocks
delete override oncontentchanged method in accountselectactivity
quickfix name should have sentence capitalization
made account jid lowercase
do not dump formatter
fix bug on sql filter
fix a bug where we d attempt to migrate depths of pending transactions
fix ci build issue on graph tests
remove unnecessary braces in playbackservice case block
add description string to constructor arguments
fix unmarshalling for pojo that extends odocument ouser and orole
used constants more
fix static aid prefix registration
fix text refresh performance issue
operator initialization does not clean state wei zheng reviewed by gunther hagleitner
add phonenumberutils method for cdma sms number formatting
include last sync time for whatsapp
forgot to remove an import
change empty string criteria
redirect jersey jetty logging through logback so we don t get log messages in the test output
add razorrepository to factory
add some docs to clockwithoffset
add check for root in correctwhattachment
replace links in markdown cell with clickable links
one more logging improvement
remove dead trustlisteners
add todo comment
renamed speak to say
add stringutils defaultifblank
add alt labels to all alts
remove extraneous param_operation
add javadoc for kill
add requestdisallowintercepttouchevent call on start stop swiping
allow serdes subclasses to access wrapperserde
fix tabs in file
remove dagger injector from the servlet context instead of nulling it
fix tutorial bug
also set tinted color for drawables set by resource
fix data info merge issue
fix the timeout issue in zookeeperstorepublishertest
do not report error on failed attempt to start external process
spring batch improved producer logging
improve a comment in recognitionservice
reduce delay between pressed shifts
add crashreportdatafactory clearcustomdata support
set a default dump type for drt as a workaround for indeterministic order in receiving dumpastext and onpagefinish callback
renamed unsafedependentmemoryaccessortest to abstractunsafedependentmemoryaccessortest
don t allow checkbox image to be squashed
avoid spurious preparing idle preparing transition in demo app
hide structure by default
replaced a thrown exception with a logger warning
add button to show changelog to about dialog
always enable fast scroll on grids
do not highlight files opened in editor
remove implementation coupling from clock interface
uses the returned relidarray from addall
shorten our menus by one more obsolete item
remove unused import
handle field name change in schema
defer springbootservletinitializer getlog
fix testdistributedqueries testshowschemas
remove a println call
add missing xmlattribute in model
don t swallow classnotfoundexceptions when configuring the namingfunction
add pre tag for documentation to render properly
remove unused imports
clarify example for deep share
override buildrequestattributes also in groovypagesservlet
fix archive wiki
avoid a call to contains
add emptiness test to avoid nullpointer etc errors in case of errors
fix not logging of exceptions in gsp servlet
missing first line of license in header
add a null check needed to use ttl in real code
fix possible npe
show the short diff in the commit dialog by default
remove dead code
set postlocation class as serializable
fix a line number on a test
changed the create code to use the same insert method
prevent language substitution
check isadded in toolbar listener
fix non responding shortcut contextmenu from arne
fix the re apply of rounding
add general punctuation regex
fix surround tests
add still rather shallow missing javadocs for classes in binaryattachmentbody hierarchy
remove duplicated method
unified config to equals ignore case
evaluate log value only if condition failed
add rsbrowser rbrowser to list
fix bug where progress bar is not set to hidden for image ajax within
adds missing settargetfragment statement to myprofilefragment
tweak constant capture
don t hardcode n as line break
getting all portlets including the static ones
only reset smsessionid if the stream is no longer resumeable
add offline node handling on reopen operation
remove unused imports
tweaked log message
remove call to killcursor in finalize to prevent blocking gc
add negative id check to material getmaterial int
make insert public
add a hidden configuration for the memorymanager grab size
add a workaround for a jira bug to be remove later as jira is being used for testing
add layouttest http tests appcache foreign fallback html to the drt ignore list
add comments to vertex
backout unneeded fix for distance update issue in searchhistoryfragment
fix nullpointerexception if extension doesn t exists
add support for the emptying out feature
enable parsing result check
add tokenstreamrewriter getsource
make datainputinternal public
add public constructors to clientcredentialstokenendpointfilter
don t increment the counter twice
partly revert cl because it broke gmail
remove line breaks
do not report diagnostic ctr in trait in delegation specifier
remove print out from test case
change caption to launch browser
should run oncomplete function when error stop tasks and remove surfaceview
exclude combined assets directory
remove unused reference to event bus
convert netty ioexceptions to unavailable
skip content in bytes
update asm in coverage
component docs do not use deprecated option
fix fc call with bad context
reduced warn to info logging for a missing spring event handler
check fields for json before treating them as normal strings
changed the logging level of gena related log messages
add missing sort overloads to transaction
don t back up apps in the stopped state
reduce path urls
turn off more_debug logspam
performance improvement eliminate multiple instance creation for
add a check to prevent an arrayindexoutofboundsexception
add deprecation warning to recid reuse
reset last editor index when insert new row
only load the dsl py when initializing the jythonbackend
inmemoryformat cached now makes use of byte comparison instead of object comparison
otr require private messaging did not set the correct policy
fix audio end listener rewind resets limit
add config exception
fix compile issue
fix race condition in httpremotetask
fix failing test due to change in jmx export default setting
add function to javascript keywords
add copyright notice to bitmapcreationbenchmark
px offset for darcula and intellij lafs
remove unused mergeevent
ensure nextstamp currentstamp
add support for preparedstatement setquerytimeout millis
prevent null ref in setapplist
fix memory leak in tests
check string do matchbracketsandquotes
add xmpperror tostring
fix bug on creating manual indexes
fix out of date metadata test
remove registry caching
fix right indent usage
revert revert skip permission and storage dialog first time
do not readd existing clients
fix overflow animation button
don t allow onbackpressed to be called while astridactivity is finishing
add overridability to coremodule
add debugging for prime connections
license should be text file
put trailing into getrootpaths to remain consistent with bbw
correct a couple comments in closurerewritemodule
add missing license header on mixinworld
catching out of memory run time exception in hasenabledaccounts
introduce simpler api for symbol tree operations
add test that revealed bug
fix the problem of drag and drop when the header view is existed
add us gov west
fix handling of error during server startup
inform the developers to close their fd streams
add date time to startup logs
return the correct register count in immutablemethodimplementation
initialize previewer with synctex disabled
improve xmppconnection javadoc
remove object null
fix null pointer in downloadindexthread
updated srtm provider url
allow longer submitter notes
restore src step
clarify the docs around the activityinfo field
add copyright info
deprecate method that will be remove
populate nonpersisted inodes
changed exception messages
corrected javadoc typo
add check for non existing path in hdfs fluid dec import
activate files pane on set working dir to project dir
fix npe in address parse
don t resolve transitive deps in filter based functions
throw error when no intent can be found
camel sjms should use same binding to from jms as camel jms does
add todo statement for master info test
hostutil gettasklogurl signature change causes compilation to fail jason dere via ashutosh chauhan
add missing break in case
fix change gighy key name
fix a missing word in appcompatdelegate documentation
add missing header
fix task leak in coordinator
remove default constructor
update paths to accuracy datasets test cases
add missing entry call
add fallback_missing to hystrix yammer metrics publisher
add segment tree testing code
expected data in vcs publishing results on teamcity disabled spelling inspection
revise cache api
add some minor improvement on scan progress dialog writes
remove diamond usage
add general error to plugin result status
beeline doesn t show spark job progress info spark branch jimmy via xuefu
fix infinite redirect loop for remote html in testapp
remove outdated hack in declarationresolver
revert fix of using compass
make felix export javax servlet and javax servlet api
remove obsolete comment
update incoming changes indicator icon when state changes
fix location url in test
fix legacy widget
randomize starting delay so tasks do not run at the same time
catch assertionerror when connecting to sockets
allow self as parameterization
add error edges to dfa
replaced action_up with action_down
hide empty state upon refreshed
remove unnecessary imports in fileuploader
use a cached singleton for primitivetype lookup name to avoid defensive copy of values array
add additional constructor for liquids liquiddata
remove package not required
improve ksdef backwards compatibility
fix osmo login issue
text editor prop tester npe fix
resolve invalid typeface style to default
made fix to prevent null pointer exception on oauth2authorization tostring when token has not been built yet
use the utility logging methods and single tag
fix potential deadlock in aggregator when using timeout completion
recreate video renderer
make analyst command line flag public
add input cardinality to global enum improved difference
create indexes on location
fix invalid cast in immutableinstruction20t
fix varcheck to not give name_reference_in_externs_error for let const declarations
disable create field fix in batch mode
fix up some fubars in mts and mta
fix a compilation error
minor add contract to jetpsiutil deparenthesize
make sure we validate cache container startmode
add android license message
fix layoutlib build
fix error related to disappearing certificates by changing alias construction
used exclusive lock even for keys and get
add methods fix in grtypedefinitionimpl
add pluginaction to result
revert remove the equal sign which caused a fc when mcurrentpage was equal to bounds size
remove outdated attributes references
add missing license header
associate the smali icon with the smaliclass object
fix bad merge
stabilize the test
fix a typo in f1 computation for the separate class
initialize default nullable notnull lists
fix crashes when remove work profile
add a new remove method
add a unit test for models implementing parcelable
throw illegalargumentexception if tx cannot be connected
add check for disposed application before document save handling
remove no op method from interface
restored api compatibility to fix kotlin plugin
fix columnindex lookups
alter cormode paper link
add additional catch in finally
add getter setter for rolesearchasuser
jglfw matches lwjgl backend target fps
remove static and final modifier scroll_multiplier
efficiency improvement don t go up the tree twice
give charmatcher apply the same treatment as range apply etc
fix the instability detection
decreasing neighbor updates to make last step faster
add parsing of cython property declarations
fix failing test
adjusted loglevel to debug
make changelogparamters publicly accessable
show saved password only if save password is checked
remove unnecessary call site dependency for evaluating return type of a property
log the ignored resolve failures so they are not lost entirely
add setrefreshdrawable method
change applied color hunk identifier to green one
fix broken test
fix javadoc comments in dbcollection class
fix javadoc warning in hystrixnetworkauditoreventlistener
improve elevation profile display for bridges patch from vivien deparday
log capitalize the bg progress title
offset magic jsinplaceintroducevariabletest testinitialoccurrenceselected
fix cs errors of camel hawtdb
remove a println statement
resolved incorrect path for readertaskfragment
fixes the get people count query
add getuniqueid to testofflinepermissible
remove unused resources variable from readerpostlistfragment
fix polygon getworldvertices
fix checkstyle violations
use double checked locking to remove sync block
remove static logger from proxyservlet
remove superfluous logger
adjust tests localdistributedexecutor for low memory settings on build servers
fix tileatlas used string isempty
fix incorrect scaled icon painting
don t go all the way to object class
schema here does not refer to json schema
fix an incompatibility not caught in the last rebase
fix toolbar in localfolderpickeractivity to select camera folder
rename libraryfacade to clientfacade
fix excludeclassinterceptors annotation parsing
android add performance data
fix problems in javadoc references
vcs reuse the virtualfile instead of creating filepathimpl on file
fix test for final fallback to blocking io
add faketaskscheduled on unit tests
fix some copy paste errors
reworded outdated comment in readerpostlistfragment
fix small bug occuring when frametag is null
set raw syslog message as full message
fix a bug
fix times in debugging output to use h rather than k
save possible reordering etc before editing names
add simple unit test on cameljaxb dataformat
don t evaluate twice a field s value when serializing java to json in objectmapper
fix non blocking queries activate db in current thread
fix bad diagnostic group reference
reverted change on logging aspect
fix a flaky test
suppress flaky framework core tests
if the node attribute of the disco info reponse is null set it to the node value of the request
reset the position on a failed iteration
add empty constructor
add wheel values to tostring for mousemotionevent
fix some javadoc errors
fix a bunch of key naming stuff
close the model view
remember zoom overview state since the postscale callback is async
don t load props if already set
add final keywords
remove unused code
make stress read lookback optional
fix another nullcheck in pluginmanager
don t clear app profiles during system upgrades
fix last commit
use of and inside code confuses the validation parser
rename test method so test data can be found on case sensitive file system
remove an unsused parameter
fix logic thanks to review by kkolinko
fix checkstyle issue
adds convenience method for some void
do not call vcs platform post startup things in awt thread we don t need it
add one more warm up run
add annotations on findall method
warning message if service not yet connected
add missing imports
fix unit test
updated dictionary needed to change test results
add ability to set text size in fontawesometext programatically
updated robolectric reset so that it calls the generated reset method
remove unused field
fix npe in tvinputservice and tvview
hide empty toolbars
change signature for javascript work in progress
fix builder issue
fix issue with fab_size annotation
add unit test function to show faulty beahvior
do not re use getconnectedclientsoperation for multiple invocations
analyze all relevant files in jetpositionmanager
made reset synchronized to avoid possible races
do not allow null map keys in the map constructor
change exception message
do not pollute cache with failed futures
fix error message
only hide grid if there is detail to show
fix orientation changes
avoid deprecated generated class
reduce verbosity of log message in businessfactorybase
fix a small bug which affects activities without any touch items
fix generic signature
cache qf counters have a wrong value for consumer time sergey shelukhin addendum
rename filterunusedclientheldcomponents method
fix failing websocket unit tests after session closure refactoring
fix a typo
reverted a dangerous change in a utility method with high usages number
add usernamepasswordauthenticationcredentials constructors needed for
remove double plural in docs
stop requestcontextfilter initializing config
fix timezone change issue when showing month name
return the created objects as per contract from instantiators used in domain object container backed model maps
zero width or height in getpicture throws java lang arithmeticexception
properly dispose completion listener in openal
check zipfile in io resource not safely closed inspection
do not run indent options detector for viewer editor
prevent preparerequesttokenactivity crash if it does not get handed
properties formatting default values
avoid npe when generating scrub error message
fix transactionmanagertest which was using incorrect jndi name
taskcontainer move meaningful tostring to the base class
disabling scrolling while refreshing has no effect on manual refresh
collect all editors from actual module instance
remove a test that is no longer relevant as ids do not get reused
copy corelabels in case the parser alters them in some way
add todo statements to color chooser stuff
updated structure view test to include object
fix copy paste error
fix unit test build failure
fix javadoc issue in operation
fix javadoc warning in xdata form
fix a rebase gone wrong with pageindicator
remove unnecessary final
relax expected exception when connection fails
moving closurerewriteclass to be an earlier pass
using concurrenthashmap for decrease synchronization
add license header to package info java file
fix npe in basestatusbar
change username to come from simpleusername from login screen instead of cn
disable testloadall by default
fix the bug that exif attributes are always cached
fix camel blueprint and camel spring refresh issue can cause namespace handler to not trigger
add proper check for framebuffer textures when doing npot scaling
fix javadoc warning
moved count command implementation to its correct package
do not disable ext dirs location because ibm jdk stores some of its essential jars there
fix cursor overshooting granularity boundaries due to incorrect gran next usage
add note to the example
fix binary map writer
put back adapter null check
fix see link in documentscontract docs
drop deprecated intrinsics stupidsync
fix method name
start the sync job service after primary boot
add constructor to construct with reader and not only file needed for url imort later
remove patch verb not supported by httpurlconnection
implemented transparent conversion between array and set
valueparameterdescriptor must implement annotated
do not call gemmanager updategemset module with null
reordered plugin registration on startup so that distributed mode orientdb also creates databases specified in server config file
improve error message
replace path separator characters
deprecate project getreuseexistingrulesconfig
update mehtod checknavbar
fix memory accounting error in operatorcontext
don t use stringutils
do not swallow exceptions
fix missing break statemetns via lukas
fix a failing unit test due to the addition of sybase as a supported db
fix rollback in masqueradingpsibuilderadapter
remove field type from result
define the default focused component
move legacy full disk encryption code to settings
add some assertions
include schema in markchangesetasreran
cannot force a shown item from an actionmode into the overflow menu
reverting html lexer changes temporarily
fix log error
rename builder method to downloader to match new class name
add application dependency injector component and module
fix backwards arguments to performdexopt
fix crash determining direction of invalid locale
fix merge bug
do not allow mapping of error type to jvm type
we don t need app
made jetcontentbasedfilesubstitutor getdecompiledpsifile nullable
lambdas predefined template
fix temp white list update in device idle
remove duplicate extraneous activetasks value
add some comments
add new options to setsocketopt
correctly validate sdk for running inspections on non java projects sdk chooser accepts on double click
add a test verifying that placeholders in replacement are fine
allocate pty for remote console
remove incorrect log message
remove dead code
print the number of subtypes
accept top level non physical classes
remove reference to fluxion
remember panels visibility state
changed modifier of some members
json enum uses name rather than tostring
remove unused arg
add more logging to pep attemp counter logic
prefix the temporary jobs created by heliossolodeploymenttest
add run on ui thread
set docstring format for test pythonhighlightingtest
do not optimize large methods
changed expected exception class to checkstyleexception
add javadoc for explanation
proxyservlet sends two user agent values
fix potential classcastexception
reduced autosave interval
fix bug that gradle continued after printing out help message
driver files download error message
flist without supports nulls
truncate cfs between singlepartitionslicecommandtest tests
sync engine send sync account missing folder event immediately
remove gradledistribution avoidsconfiguringtmpdir
fix renderer count
add hints to javadoc
fixes add remove correct listener for opsetsms
updated config and messages
domain directory defines now correctly only in grails application
fix copyright headers
java use the correct offset for the channel when logging
prevent iterator from being called twice
fix typo in opcommand
fix javadoc reference to contactscontract contacts
remove redundant null check in wrap
add back the governor labels which had gone missing from the treegraph
fix an issue where post page preview are not shown in viewpostfragment for http protected blog or self signed blogs
made two strictmathtest tests osx only as hypot using nan fails on ios devices
fix a comment to fix the build
remove todo after talking things through with daz and realizing that it s not a real problem
remove unused helper method
fix up reversed assignment for version fallback
fix some javadoc errors
add youtube link on text justification qs
add onreset to compasslistener
add missing since tag
fix a concurrentmodificationexception
fix bitmap load from stream resize options
allow configured ports and initial token to be overridden by system
imagedownloader uses thread pool
fix availablecapabilities incorrect type
add event for show number
skip tests that do not work
remove unused function showconsoleprogressdialog
allow raw image widgets to extract to chunk output gallery
don t propagate level or rtl in layerdrawable setdrawable
fix android manifest merger from using only one mergee manifest
enforce provision check for csim record only
break the loop on connection closure
avoid cce caused by jvm bugs
add javadoc to groovytargetelementevaluator
add recovery kmsg to dropbox
fix possible index error
run gen thrift java to update thrift constants java
remove unneeded jars from logging legacy controller test classpath
update start immediately navigation
add a todo
prevent splitter from going lower than splitter height
remove commented out code
disable testtransfermigratedsubscriptionwithctdinfuture for now
remove redundant state
fix javadoc for hystrixthreadpoolproperties setter
fix quality flaw
fix issue on replication
use double precision for double with firebird
add missing copyright header for whitelist java
prevent classnotfoundexception when restoring viewpager
fix a flickering when expanding and the notifications updates
add suppresswarnings unchecked
mediamuxer sample code update no functional change
remove unused import
collection binder corrected a bit
remove throughput related methods
revert fix commit issue
remove assertion in editorcomposite
undo the debug things lost track of my branch merges
using a messagedigest per hasher not per messagedigesthashfunction making the latter thread safe
modify handling of text antialiasing setup
fix the other sentence s expected output for annotationoutputteritest
remove debug printings
stop exchange operator immediately when task output is done
don t process remaining dispatch requests once the socket has been closed
continue loop in signingidentity
fix pipe renderer using wrong texture for center
expose apple platform name to skylark
addressed code review comments in junit38classrunnertest
remove unused imports
enable copy urls pop up menu item for all messages
pass disconnect reason for failed remote connections
fix bug in short cf names
consider java time zoneddatetime a swagger datetime primitive
don t pass merged properties to buildformdescriptor
add getter to the user object associated with a base adapter helper
add account name type query parameters for calendar events
promote guardedby from experimental
demote the log in processstate ensurenotdead from a wtf to a warning
change newinstance to a generic method
increase the amount of max possible folders
add keyword try
shorten long messages so mysql doesn t stop on insert data too long
remove unused imports
fix matrix issues
get logic right way around
do not return xml rpc endpoint found but have missing methods when the server replies with an empty document
add a comment that describes some gotchas around writing options file on windows
fix interface break constructor change
move to data flow issues group
fix deprecated invocation
pass editor to release when releasing deferred missing editor purge
remove needless copy constructor from deleteindexedscriptrequest
add util jar to classpath only if our urlclassloader is used as system classloader
remove unneeded read action
delete some code
fix some more license headers
scope view panel should ignore changes in injected files
only list unallocated shards on data nodes
simplify get by id
fix a couple test failures with the new event code to trigger existing authenticationfailure logic
update javadoc for settings secure android_id
transitionset setstartdelay did nothing
infer some punctuation tags easy mode
add accessor methods for editor status properties
comparison method violates its general contract
fix world region reports url
do not show an error hint in unit test mode
correctly detect cmd click on mac desktop
add support for a back off multiplier capability to the scheduledpollconsumer
add support for application atom xml and like content types that might begin with application and end with xml
don t show icon for power off confirmation dialog
move test field at the top of dialog
zoom should not change sync with map
uses string comntains instead of a regex pattern
always execute actions on the runqueue
ignore unknown filters
increasing logging level for when websocket fails to open due to exception
changed power request to activation energy
avoid double slashes in path names for camel ftp
add new font awesome icons
remove unused imports
improve fakefile implementation
add patch from peter dobratz to improve documentation
fix infinity loop in media previews when the icon in the action bar up button is pressed
delay instantiation of datasource as late as possible
remove test checkout dup ooo rows
display client version as well
fix build break
tostring should never return null throw instead for invalid state
add support for svg images in html output
use a built in marker class for configure kotlin action
handle illegalstateexception from updatepuller and convert into a comexception
android sdk fix placepickerfragment search text timer to avoid null reference
corrected json socket runner test name
bypassed problem let db to be open
fix race condition when run is called concurrently annotating the idempotence
fix previous merge
fix the big seekbar setprogressdrawable bug
throw illegalarugmentexception instead of ioexception
fix nashorn test
fix bug with password storage where some characters would lead to
prune unreferenced contents faster
revert change in http resource cache
cometprocessor should extend servlet as they have the regular servlet lifecycle
add a command line flag to enable using types for optimization
disabled a failing test with a comment to investigate why
annotation parameter info should select a current attribute if the caret if between it and a right parenth
got back the string constructor for kotlin compatibility
add classlocation methods
remove warning crap
remove requestfeature from activitycontroller
fix comments cab by switching to appcompat version of actionmode
null checked additionalrenderstate when loading a material
remove view s long press callbacks when action_cancel is received
restrictedlockutils introduce check for suspended apps
really fix the findbugs issue this time
remove validateschema from deployment
rsv ordering fix
psf fields with primitive types should have correct value
add javadoc in world
fix null ref
remove supresswarnings in viewfetcher java
promote an info message to error
fix problem reported by checkstyle
treat thermostat mode set as a report
made creator inner class public to accomodate testing
implement date literal
changed iosmethodbinding overrides to returns true if the given method equals the internal delegate method
remove unneeded override
don t add the dump heap on out of memory to groovyc unless asked
remove meaningless javadoc
make lock task check if focused better
ignore synthetic methods instead of using presence of in name
add test for plistserializertest serializetostring
force fake filesystem to behave like unix
add docs for availableprocessedsizes
adjusting method call
add missing ownership entry in hiveprivilege parseprivilege
make formattimestamp methods private
add escape handler to darcula tables
do not log runtimeexception
deprecate support for ganglia output from metrics
fix memory cleaning of global variables
add small log debug output
add expression string to failed expression evaluation exception message to help clarify the context
add some more debug trace
add a small gap in front of line numbers
activates javautilloggingconfig used in android
remove places where db instance was not required
fix checkstyle violation
fix inconsistencies between yaml and code defaults
fix the failing vptreenodetest
only add the coregameplay module if the mod selection is empty
securerandomsessionidgenerator allows configuration of the session id length but doesn t actually use the configured value
add constructor with size
activate terraingrid in sdk via jmeterraingrid node wrapper
fix chunking issue when write is called with empty buffer
remove exit button from sherpafy
hotfix remove recoverymode from jobmaster
fix typo in javadoc
add additional test testinsertonetomany
check user state after clearing identity
for unrecoverable projects rename them with a suffix so the next time we won t try to recover them again
add optimization for like usage
don t get documents for files with read onloy file type
renames yolo suggestions because they re not yolo
ignore show content events in source satellites
cleaned up messagebarutils
handle error when user closes browser while playing snake game
provide a constructor that captures the json
don t cas on exit
allow multiple notification of closesessionfuture
remove obsolete fixme since now the chatconversationpanel refactorings have been merged
switched method order
fix a npe when searching for rooms in the searchchatroompanel
fix concatenated gzip files in statics3firehosefactory
use the same typo in both saving disco and reading disco
reduce number of rows in multivaluedictionarytest it s killing travis
fix serviceaccountcreds test and computeenginecreds test
remove catools dependency for shiny
fix getconfigurednetworks when wifi is disabled
add verbal alert description to alert email templates and subject defaults
remove extra latch countdown that can caused non deterministic test failures
map reduce test
implemented missing getchildtype method
remove debug output
fix bug in fragment transitions where shared elements use same name
adapted junit test to new serialization format
fix annotationmapper as well
handle npe when no changes in backing index but a changed unique constraint
fix issue on record null sent by the server to the client
remove shuffle on selection of the nodes to open
remove unused mocking in the test setup
set visibility of top components to package
remove fix todo task
fix broken informationschemaservicetest setup
fix failing test
ignore connectivitymanager leak
add getserveraddress to singlechannelmongoclient
ignore tab character in contact list search patch by markus kilas
add some new useful short methods inside the iconcis class
fix pieae in referencing obsolete cached psi elements for built in str unicode type
don t pass a null view to onlistitemclicklistener
remove unnecessary exception handling
fix new javadoc
implemented basic version of open path buffersize
postpone user cleanup to systemready
fix html char conversion
don t cancel any notifications that are meant for all users if no package is specified
fix onrequestpermissionsresult scope
propagate arrival delays from first stop
and now also set it
add a missing else block for throwing a store def mismatch exception
set kw_compaction back to compaction
deprecate the defaultreadonly property in favor of readonlybydefault
fix typo in image mime
fix transient failures by scaling wait periods
correctly handle interestedops changes
consume exception if native library could not be loaded
allocate correct buffer size for search query caching
jspx supports jvm debugging but not in xhtml files
updated realmchangelistener javadoc
create parent directories and fail fast
add blockflower to softblocks
fix line endings in dependency injector generator
changed the close method to be private
skip duplicated urls
remove wildcard import
de fix false alarm um ein uhr
statfilter add getter setter for logslowsql slowsqlmillis
shorten if else statement in offlinemessagestrategy
fix issue with framed channels
fix findbugs issues
find usages rename css in jspx
fix database open in case of server database path not default
load a truststore if client certificate authentication is enabled
fix test for java to kotlin copy paste java standard classes used to not be resolved
add abstractblockchain removewallet
portlets are tracking their own instance
fix crash when the device is turned while the warning dialog about server certificates is shown
use the flag from class as the interface abstract flag in inner class
custom tag noinspection cause build problems during release
persisting object instance instead of proxy
add a d keybinding for delete
linkedin authorize url changed
implement filesystemmaster getprocessor
renamed restrepository to restdatasource
fix loadmetadata initialization
do not throw webapplicationexceptions in inputs java lolwut
remove unnecessary shiet
fix a bug found by find bugs
add license header to ifiledownloadservicehandler
fix inmemorytitangraph change
remove deprecated code
fix astrid up button not working in landscape with edit view
increase agent connect timeout for slower loaded windows machines
fix ml types in cli
fix filters execution order to be from lowest to highest rather than the other way around
bottomsheetbehavior passes view to callbacks
fix a bug that caused task edit activity to quit on rotate
hook up nextmessage to ack mutation
add info messages on incoming connections as well
configure the job jar
goconfigwatchlist should listen for changes in file config
manually remove spaces as add between import statements
replace hardcoded with tachyonuri separator in concatpath
avoid illegalstateexception if called on empty selection set
fix possible npe
fix displayed sentence
remove mex connection constraint
sql editor paste fix extract from source code bug
messagemanager remove active chats on account disable
add mjapi getenergystored and maxcapacity
install dependent service after dependency to avoid intermittent missing dependency reports
remove dead code
fix crash when doing conferencing over wi fi
add additional constructor for liquidtank
increased space between merged messages a bit
fix command build bug
add ksmetadata hashcode
fix refactoring bug
use correct messages
add hash to metadata print
scope chooser has unclear options
fix linestring constraints
need a multipart config for javaee containers to handle multipart at least sometimes
avoid exception when disabling new rendering and closing existing open editor
remove whats new dialog
add getchildcount to pshape reference
changed control scheme
fix rounding error in seekbar tickmark drawing
fix rooannotationprocessor metadata
add check on start up
enable dolexingoptimizationcorrectioncheck for a while
remove new asserts in cmfd
remove comment after retesting and finding it now works again
remove dead code
fix clustered node startup problem
add a tag to separate paragraphs in javadoc
exclude file in config quick fix not ideal implementation though
resolve issue for integrationtest by add missing packages for arquillian
remove unused satellite window import
add a deprecated annotation to runfiles addartifacts
refresh list view more often
fix theme change not applying to showsactivity immediately
fix issue with new extractor input
print full classname inner class support and fix enum output
run plugin s configuration properties
have the notification service only have notifications diabled
remove invalid characters probably related to a merge that went wrong
fix wrong argument of standardwrapper remove in case when singlethreadmodel and instancepool are used
monitor the right interface re vpn
fix another spot where we were comparing strings with
add missing author tag
add missing override annotation in abstractmessagetask
camel component docs now include a human readable title for the component names
replace usage of unknownhostexception with unresolveableaddressexception
fix extra document changed event
making alphanumeric alphanumeric
simplified always true condition
fix merge issue
migrate users from c g c base objects to moreobjects
updated pgl android
in form preview mode ignore bindings for button groups
make the constant final
revisit error message when loading the catalog
remove project verification
remove unnecesary breakpoint requests recreation after position manager append appendpositionmanager usage javadoc updated
changed property name use for clustering enable disable
forgotten annotator was breaking some tests
fix debugger disconnection
add more tweaks to flaky testautopayoffwithpaymentfailure test so it becomes more reliable
fix semantic of defaultattribute impl on setifabsent
add the rather crucial missing i in the wait loop
moved static final variables to form class
add more logging to searchpreferencetests so that we can know which preference is failing
do not fetch tasks that don t have activities
speed up pathkeyfilestore by using singleincludepatternfiletree instead of directoryfiletree
fix the top gravity support and dimming
fix bug in qtestutil where lines after a partial mask will not be masked thomas poepping reviewed by sergey shelukhin
document thrown illegalargumentexception
add final modifiers to threadpoolconstants
remove fields incorrectly pulled in merge
remove another configurationexception which i overlooked in till s fix
add missing copyright header on cursorexception
wi surround seelction when quote typed made an option and is off by default
add support for mariadb in cleanuplistener
throws redisexception for rbatch execute method
add a small test
fix screen rotation animation
fix quality flaw
set the accept and content type to application json explicitly
fix crash on boot
cleanup remove unused imports
add getcontext to fakedialog
fix wrong link
reduced usages of stringutilrt
fix a typo in a getter method
deleted unnecessary call to wordpress getblog as per nbradbury code review comment
don t change the log level was for debugging purpose only
fix domain xml marshalling bugs
fix bug on polymorphic ids
add test for date vs datetime timestamp field in message
do not throw unexpected runtime exception to app
fix javadoc make constant all caps
extend test in iswardeployed to work with weblogic
fix typo for magic links
improve error message
avoid starting services to soon during camelcontext startup that can trigger circular dependencies issue with spring and similar ioc frameworks
don t create synthetic entries for excluded output folders fix importing excluded roots from maven
fix open bug in facebook
setting notification content text otherwise it would only show the title when collapsed
immediately return existing world if any
make heliosdeploymentresource before after public
fix intermediate points dialog
remove bks comment no longer used
add the cleartool command
update demo for new setemptyview
fix npe in lwjglcanvas restorecanvas
don t set path home for integration test
don t specify tags in runner
fix a row multiplication for sparse input
add some missing javadoc
set the state to cancel instead of resume
fix cs issue
test create standard restclient at first request and reuse it
pass down the engineconfig to indexsearcherwrapper
reverted back templates folder setup in django
dispose indicator even after exception
reduced too broad catch block
add regex matcher for uris
add netrc support to gitclonefunction
fix license header
add methods to convert an immutable modulecomponentresolvemetadata into a mutable mutablemodulecomponentresolvemetadata and vice versa
remove a todo note and add key count implementation rough for now
fix issue with finding duplicates
add extra logging
always update drawable container source res
add hashcode method
add convenience overload of userinput that takes list string to make it easier for scripts
do not show weird symbols in console
rule getrulescategory must return a fake category instead of null
torod core add seriaversionuid
fix wrong syntax for mandatory test on create edge test
remove unnecessary digester configuration
add percentage contended
add support for using inside a fragment as fragment children
add applylr to rntn
improved compatibility check failed error message
add doc on the canscrollhorizontally method
fix bookmark text
change color or background for set from
fix another regression
clarify behavior of responsebody string in its javadoc
define action used to show regulatory info
fix dx checking on windows
add some todos
corrected castvec3 method to set value instead of name
fix two minor javadoc typos in getter and value
don t use private sun api in jscontext
add test for chain insert
add comments to indexedset
remove predicate function from taskstate
remove debug artifact
use a single geometry factory for all geometries
remove short_string_max left out of the previous commit
properly initialize triptimesubsetcache when graph is deserialized
fix static setters for arc
add stringutils repeat
fix junit for instability detection
remove unused import
fix wrong scaling of compound drawables
log debug error with buffer pool loading
remove e printstacktrace
java fix warning
fix static import warning in prestos3filesystem
fix the nested preferences title
replace link with see
change rejectalltestsfilter to only reject atomic tests
fix issue with blockworldvector causing java lang verifyerror due to how bukkit loads plugins
changed exception type in prepareviewsforanimation method
associate the mediafile with newly created image span
loadkeyindices starts a transaction but does not commit it when the graph is being initialized
still honor the false value
set claim sequence when the claim happens and not on publish
add shortcuts for elements with no annotations
improve comment why we can change to
fix an issue where self hosted blog could return invalid remote blog id
add deprecation information
remove lz4fast from compressedobjectstrategy for compression since it is not currently used
don t show python skeletons in python sdk libraries inside project tree structure
add another failure condition
rename metric ws searchaction variable following java standards
add network check before liking
temporarily rollback a change that s causing setupwizard failures during restore
fix typo in javadoc
fix pointer issue
fix bug concerning file channel implementation
add name attribute on test suite
add frameworks base changes for enabling reduction proxy
fix a crash with singlepass an multipass tdl when a lightprobe is add to the scene
add more unit tests
fix some errors if the config is unparseable
remove test testinnullwithnullargument
add a proper comment for sortactivity
made commandfailureexception public
move mcontext into webrenderer
make scroll to selection in file history work better
add spacing ftw
restored static imports in observable
permit forwarding from a future v to a future
fix logic error in downgrade of system apps
register the default healthcheckregistry as default
fix mining well not calling super when placed down
forcing software in header button background ripples
fix potential npe
remove unused quick setting display preferences
avoid npe when password empty in jcr endpoint uri
fix issue with ajp request stream
smallfix add a todo on master side
fix http patch application start message
test don t print bwc test path it s different on every machine
fix a bug in hide and show panel
fix drag to add tiles going away
once again fix a flaky decrypt test
fix for wrongly marking as used thanks to alrikg for finding this
fix tree issues
use setter for worker last updated time
debug fc when route is finished
create percentlayoutinfo on demand if not already set
do not merge small documentation fix for audiomanager java
changing thread name to contain version info
add a couple of test cases
do not inject null
prevent pre commit check runtime exception from killing to parent commit process
don t sort primary blog to the top when searching
preserve pre existing surround behaviour tweak display text
fix my changes which broke request desktop site functionality when user had specified desktop as global default
remove a log
fix mappedcontainer deserialization
fix jvmlibraryspec dependencies javadoc
some openssl aliases use edh instead of dhe
made decal constructor public again because gwt explodes otherwise
use this instead of class when comparing
submenubuilder getrootmenu should return root menu not parent
hide notifications window automatically if no more notifications left
wire the test task to the jvmassembly rather than the compile task
add allbindingschanged handler to initialise polling table
improve templatesend method to check the response code to see if it s successful or not
remove some redundancy
browser actions linux fix
dismiss recents when pip moves to fullscreen
use separate tunable for thread pool size for handling renamed files
add stone and cobblestone to ore dictionary
fix redundant string constructor call
just check everything for null before you do stuff with it
handle disconnect in middle of dhcp operation
fix log on cardlistview
linker didn t replace lookup functions with synchronized wrapper function
add proper type parameter to constantimpl usage
deltamanager extends lifecyclembeanbase so it registers itself
remove logging of values which might be null causing npe
oracle xml tables reading support
don t fire selection change event in prelayout
add one more status code after which we re trying to login
reorganize fault test
fix episodes not getting flagged when flagging a whole season
revert incorrect changes
remove flog isloggable string int
fix bug w dissoc on hashcollisionnode
pluginbuildsettings refactor broke layouts views from plugins this fixes it
implement fileutils contains
smallfix fix style
add todo to remove cyclic dependency
add one logging output
tests decompiler performance test timing too optimistic for tc agents
add some printing code
remove inaccurate todo comments
print command errors to console stderr instead of to the bare stderr
fix javadoc warnings
remove code for debugging
remove the todo and add a comment regarding the mix usage of android app fragmenttransaction and support fragment
prevent ise at broker from completely failing to return results
fix treeset s comparator
fix the cs errors in camel core
specify known size at list creation
add snippet macros
fix the npe of jparoutetest
add status code javadoc
add support for expandablelistactivity
add new constants
fix javadoc warnings
remove empty line
fix npe in colorrgbadialog introduced by previous change
add method to get variant for trip
add a long description for some processes
naming convention fix
log uid and pid of legacy route requests
get post rename callback before do renaming to avoif invalidation
core properly handle empty layout in getfirstvisibleposition
commit message tooltip hidden in file history tool window
mimic apr behavior and do stuff according to document timeout should not close the connection
updated vcard test to test illegal xml characters
jetty start add enable an disable functionality for modules
fix bug in find conference in database method
fix a bug in the rsdr program
don t allow renaming nodes not declared in sketch
fix potential npe
write device owner name to xml
remove unnecessary code
fix pm crash due to bad array bookkeeping logic
add disclaimer and example
fix possible npe
skip preparing the socket read buffer and invoke directly socketwrapperbase read boolean bytebuffer
combine result of both listeners to determine if the collision should occur jbullet
add missing constructor parameter to keyevent
fix multiple calls to firechannelactive when using aio
improve ws rule search example value of qprofile
remove hardcoded package name ejb jndi util would come in handy here todo
add references to other constructor annotations in the javadoc
install and update buttons only visible when net is active
use cache streams to copy over byes not the byte array
fix quality flaw
add raise_fee to transaction purposes
don t save anything by default
fix the output format for the
niodatagramworker should reject the write attempt on an unbound
correctly handle earlier versions of websockets
add missing javadoc
fix wrong translation key unnamedstreed unnamedstreet
fix trivial typo
handle the default modifier
fix selector used for the isplaying method in oalaudiotrack
test use more verbose assertion in indicesrequesttests
layouts restore state after a layout pass on the scrap list
move setthreshold to initial setup
bring back warnings
add test for topmost home stack activity being home
remove voice search constants factored them out into the voice search app
tweak parser header col detector
fix hiero clipping glyphs if a glyph is larger than the page size
fix the fix of the fix of javaplugin getresource
add magicconstant javadoc add
avoid double call handleexecutionerror
add test to verify protocol relative urls
fix previous commit cleaned up unroll handling
remove extra integer ctor call
fix trivial mistake spotted by benjamin bentmann
file should be selected instead of resolved element
broke this test
create action intention should create action in bottom of controller
allow change in whether to shuffle training data after checkpoint restart
fix client error handling in asset handling
simplify return value in validator
socketioserver stop causes npe
override attachableimpl tostring
fix an eclipse warning
include entire exception
go to action don t put separators into the map
add setter to prevent standby
fix missing cherry pick action
fix full screen activity from showing up on rotation
add support for specifying broadcastercache
clarified javadoc for scrolluplist in solo and scroller
remove deprecation of charactercontrol
fixes the appearance of the contacts group
remove unwanted findbugs supression
fix wrong class filter condition
safely notify cluster state task executor of publish
anna do not wrap declaration statement
don t convert file url file
do not start compilation for disposed project
fix dialog shadow
remove unused call
make checkbox opaque if it is root of rendering component
avoid unneeded synchronization
corrected wrong package name in javadoc
only save cpp docs before sourcing if they are dirty
fix isthemeappcompat npe when using an invalid theme
log out of band caching
add shutdown of httpclient to ondestroy
reading the newline characters
add test for different number of days of cleanup
adapt to latest changes of osgitypeconverter constructor
make a var public
introduce actionplaces toolwindow_title
fix private field
fix a cs error of ahcbridgeendpointtest
don t display the pattern in the dialog for now
prefer user values to maintain consistency altitude
simplified list creation
fix class cast for errors
rsv segment read fix in disconnected editor
improve clarity of clearpersistfilestest
remove wasteful default instance allocation on deserialize
avoid nullpointerexception when trying to move a message before it is fully loaded in message view
add comments to new classes
remove deprecated operationservice dumpperformancemetrics
temporaly remove status message for the first status change to see will the build failed again
release wakelock on ril request send error
fix symbol to qualifiedname conversion
remove redundant this keyword
remove dead commented code
create a new courses directory if could not find one
fix build problem due to changed constructor format
fix sybase ase database identification string
updated example configuration in javadoc for ldapauthenticationprovider
fix servletcontextimpl getservletregistration bug
test temporarily silence failing bwc test
add sender s address to warning on joinrequest to be able to identify it
remove javadoc warning
xpub verbose mode excludes unsubscriptions
fix npe caller may not pass loginoptions
remove that part for now as it suffer encoding issue
somehow some merge garbage got into the repo
fix class cast exception support info
remove ignored tests
fix reference counting of wifi locks
remove dependency on bootstrap
turn off debugging in multiwaveview
remove duplicated code
revert timestamp only change
fix bad logic checking the result of mkdirs
fix checkstyle errors
mark just this method as slow can easily take over a minute
explicitly set the default value of the field dashboard global
fix bug in jsr web socket buffer limit code
fix npe in tests
add default empty constructor to rateoptions
use append instead of when concat strings
remove duplicate slash from resolved webjar locations
remove unused imprt
provide default directory
remove newlines from error message about fonts
use weak references for lock keys
remove package private constructor
do not sort completion variants using matcher with empty prefix
remove mis placed broken and obsolete path restrictions
don t use peers that claim to have a block chain and then report a height of zero
add kotlin runtime to classpath compiling java sources for tinyapp
set deferring locks true by default
fix snapshot filter for frame
minor delete kunit source root from stdlibtest
removes accidentally committed system out println s
properly load the cometsupport instance
remove proper arg when dynamic true
add support for mutation of results e g with
add import for locale
add debug logging with effective value for iodotnetty leakdetection acquireandreleaseonly property
clarify what storeversion is
add test of backpressure to operatorall
disable hard consistency check
remove useless return value
modified sleep value in dialogutils
fix null pointer exception in config getmaplist
properly negate these conditions
fix broken backup path on windows
fix a small bug where an npe can occur if a closeable is not present
handle minor timing differences
fix template offsets in docstring generator
add some javadoc
encode textual bodies as quoted printable by default
file system tree autoexpand root node
remove extra space on entity declaration in the main docbook xml
implement constants is broken abstraction use import instead
mark opened files as dirty before refresh
minions in las now get a name
remove meta classes from registry on teardown to improve mem consumption
screen blend mode for andriod
accidentally committed rubbish
fix ui feedback
delete optional in data media media class
fix failing test
remove unnecessary log warnings
fix crash support info
create inspection xml if no problems
cannot throw exceptions from reset must fail the callback instead
add a bit more logging for the daemon criteria matching
equals method modified and hashcode add to activitylifecyclecallbackswrapper
do not lose focus on hiding of properties panel
jetty start startup incorrectly requires directory in jetty base
remove unnecessary collections unmodifiablemap system getenv and environmentutil getenvironmentmap are already unmodifiable maps
add es hostname to es_open_files notification
fix issue with destination point description shows multiple times
fix compile errors
moved slow tests to nightly
fix npe in style fragment parse
suppress deprecation warnings on the implementation classes
simplify escapexml object
fix reversenestedaggregator to compute the parent document correctly
add checks for unexpected tokens in parser
remove redundant isclosed check on tryclaim offer
add version of butterknife findbyid for dialog objects which infers the target type
restore ctrl q from completion and ctrl n
don t record changeshappened for featured lists
fix munge handling of
fix warning and add perf test documentation for pingponglatencytest
handle activation requests as authority only
remove redundant assertions
remove obsolete import
fix checkstyle issue
fix import bug
add validation of alpha parameter in glm
fix entity parser
correctly read the lines from
adds braces to if for single statements
add supported refresh rates for all displaydeviceinfos
update a misleading outdated doc
fix notification bounds when top notification is scrolled
remove unused constant
use server defined default limit for get lists batch size
remove a debug message
changed mesh restored indices buffer limit
implemented inventory contains with a minimum amount of item required
hide internal class from users
add some javadoc
fix entitlements substitution documentation
give imes the entire screen to use for measuring purposes
fix class imports after rebase
remove the leftover imports
add indexing example
do not require bbox parameter to geocoder
fix wrong test
fix service binding for singleuser services
replaced full reference with import
disable flaky tests
fix checkstyle issues
improve documentation on vibrator vibrate
capture all structures of type collection
remove an unnecessary suppresswarnings annotation
use new result type
propagating indexing exceptions properly
don t save history in tests
add animatedtiledmaptile getcurrentframeindex
for better handling of some formats perform rotation with help of graphics2d
fix reportlostfile call
add missing publicapi annotation
ease the validations on databaseconfiguration url
don t reschedule syncs that failed with a toomanyretries error
make nullable contract visible
updated createcollection javadoc to define capped collection size in bytes
more specific run configuration icon used in sub menu
remove additional incorrect javadoc from userdetails getpassword
add hxx as filename extension for c headers allow to configure extensions
reset next full trakt sync when connecting to trakt
clicking the menu button crashes jellybean
fix deserialization problem
fully qualify javadoc links and make the tools happy
add thread total started count to systempublicmetrics
use proper type eval context for parameter info popup
fix the typo of the filecomponenttest
commented out test until passes
uncheck the generic converting
suppress strict mode from calculatebestthreadcount
remove unused persist radio imsregrequired
add pivot method to tableorview interface
add method to customize search view for light theme
remove unused method
changed cascade for many to many to save all as the sensible default
prevent npe in login with null username pw
add more javadoc
try slightly more useful error log in nioclientmanager handlekey whilst still keeping it concise
remove no longer necessary line
also unit test the creation time of a fresh key
adjust the mapping of lte signal bars
call systemready on the lock settings before doing so on the window manager
fix failing tests
capitalize dialog title
can take a path argument
large value formatter now can appends a specified text to the result string
provide a better error message if process is called without the processor being started
fix singleuser provider grants
fix outdated pronoun disambiguator test
fix issue where existing activity resources references would not update
do not switch view during package navigation
fix no type to search
fix flag_dismiss_keyguard not working
switch default security filter back on
can t make assumptions about the exchange anymore
cleanup compiled configurations on project close
fix networkownershiptest unit tests
remove maximum limit on archivedfilecount for file logging
fix broken test
delete enries from ofuserprop when user is deleted patch by peter johnson
allow motionevents to be cloned
notify chat closed when reusing chat activity
add ready method when madvoc is up
serialize expressions true and expressions false as boolean literals
accept optionvalues as task property annotation
when remove asserts preserve the assert tightened type on the remaining expression
remove unused code
changed providerjars from list to collection
fix sap sybase ase identification string
fix problems following rebase onto master
save files before check
fix the flash caused by missing setmatrix when surface size changes
add missing flags field from goaway constant
closes completely the master password dialog when finished instead of just hiding it
fix flickscrollpane tap yeah my previous fix was wrong
fix a lint warned bug
call super for consistency even if it is not
change visibility to package private
remove commandcallable getvalueflags
synchronize sipwakeuptimer onreceive
just fix a typo
enter handler fix
add two type hints to make eclipse ecj happy
don t overwite query parameters in requestdispatcher
commit no extra vcs refresh
add snippet macros
add missing fluent builders to dlc for java dsl
remove unused method
wrap the logging
remove unused interface
fix logging statement
cleans up threads after rwlocktest
persist note on update
restore console default statements exec
fix system getenv test on ios
fix some javadoc formatting
found a small bug in kdtree
quote all lucene query args that have non alphanumeric characters
do not pass panel of already closed dialog as owner
add class javadoc
remove debugging code
add method to shutdown the server
appeasing the check style gods
stop prefilling self hosted url field with empty string
fix constant typo
add missing synchronize when pushing messages from mam backlog
simplified implementation by remove template method
fix return type of querycacheconfigreadonly setcoalesce
fix unused import
update scan results when frequency band is set
adjust chart height to accomodate the newly add serializers
add group by having order by subquery tests
make one more method protected
add method to return raw message object
fix npe when handling routing failures
fix checkstyle errors
increased side margin for showing docked stack dim layer
fix the sdk build
remove private constructor
remove extra break statement
exclude incompatible test case
ignore flaky test
fix random typo
fix incorrect param names
remove unnecessary constant
set auth header to make updating sig on replayed req easier
made sync logging slightly more readable
remove assertion in hot loop
fix npe and problem in which custom authenticationmechanisms aren t used
remove no longer needed targetapi annotation
add permissions for the cluster config
fix misspelling of in last commit
fix the build missed a renaming
retrieve suppression actions in template languages for embedded tokens
send migration completed event after sending partition runtime state
fix saml tests with matching issue
fix com alibaba json bvt bug bug_for_smoothrat8 to not rely on unguarenteed order
add float symbol
made redblacktree and hasharraymappedtrie package private
change period to semicolon show javadoc shows it in the one liner for package infos
adjust log level in loggingwebsockethandlerdecorator
remove warning so that titan silently ignores passed in ids
when pre emption is disabled task scheduler gets into loop prasanth jayachandran reviewed by gopal v
fix bug with incorrect output directory preparation
disallow switching managed profile to foreground
fix invalid conditional check
remove an unused import
fix the useless warnings about could not generate link for unknown class t when generating docs
dependency parser should provide dependencies
do not append test directory path with
improved indicesoptions javadocs
add a gbm flavor to logging
deprecated debuggermanager registerpositionmanagerfactory
corrected package name and also changed the argument
get correct target platform for codefragments
add a display name for each contact source
add missing license header
make python completion weigher work only in python contexts
remove unused field
refactored two more private constructors
remove unneeded log
provide way id as help information
adds a comment which explains why we pick the first role from network response
don t reset row height
use the pooled allocator for performance reasons
fix robospicecontentprovider to declare single item uri type as item rather than directory
add destroymethods to datasources
dispose path cache for sdk when project is disposed
deprecate the autovalue
make names consistent with standard java names
rename most of the parameters that had key in their names to be more descriptive
fix crash when trying to register a trakt account
remove jetty info from http headers
don t start information inspections e g if file belongs to cls
fix varargs tests
fix bug with bad state in animators
set us locale for tvdb date format
missing key on file export
remove testselectrandomtwotimes in transportsqlactiontest
fix a bug with tag context menu intent names
ignoring bad canonical uri alias
getwindowvisibility instead of private attach count
fix comment bug
have getphonetype return phone_type_none for non voice capable devices
move stack dump off of inputdispatcher thread
handle literals in output columns
prevent npe in coolant api
remove redundant voldemortconfig allocation
remove decompression code in compressedsequentialwriter
add missing license headers to be squashed
use a default resolver with bootstrap resolver null
forgot a minor bit the alt impl
fix a concurrentmodificationexceptio
fix an npe in searchdialog when clicking on mic button
updating some of the wording in the installer notably any hint that this thing still does anything for netbeans
fix connection closing behavior
add system property to allow users disable native fs for windows
remove conference window scrollpane border
add a helper method direction opposite
add arguments to api breakpointinfo
fetch user from database instead of using the one in the session to avoid hibernate exception
reduce logging noise
fix comparison of double value with zero
remove one ignored test
tidy up remove unused variable that was incorrectly left in a previous change
check boot permission only when parameter is true
close the buffered source reader
changed onbind method to use getmeasured size
changed clienthttprequestinterceptor array to list
allow subclasses to add their own custom grantedauthority s
alter table drop partition is acquiring the wrong type of lock alan gates via ashutosh chauhan
clarify javadoc for securitylog
remove stray debug println
fix upload media error
add wait for nodes to restorepersistentsettingstest
add index name to the log message
clear the google token to ensure it isn t stored
fix test case following change to mapper
terminate dd dt auto closing at dl
add ability to remove hints from answer placeholder dialog
remove log statement that was accidentally left in
move registration button on front page
don t allow resize of param dialog
fix compilation errors
add explicit nullable annotation
fix export action enabled problem
moved connection check in class outgoingconnection to separate method
diff draw shadowed lines instead of simple ones on gutter
fix exception when reporting disconn cdma state
updated precondition message
add test to classname use property instead of hardcoded false
fix issue with relationships in json db export
remove forgotten debug output
print some more info
on tablets don t do a full refresh of the task list just requery full refresh deletes temp tasks
add a constructor that knows about named
improve type casting for jsonpointer methods
update of class comment
fix npe in creation of root folder
fix constantly mipmap generation
supported supportsthreadisolatedtransactions to avoid tests with reuse of graph instances not supported anymore
remove misleading comment getsystemtray actually throws that exception earlier
vcs don t treat a root as unregistered if its vcs doesn t support root checking
fix closeguard reported issue in wiredaccessoryobserver
fix a filealreadyexistsexception on unix systems
add a getimageurl method
fix guimanager permissions
reduced defaultlistablebeanfactory s override logging to debug level for equivalent bean definition
add delete users resources
don t spawn thread to wait for sendworker on handlesilentexception
fix cnfe in small ides
fix shader name
remove the line that manually set the response status code
fix minor typo in android content syncresult
send the close immediatly rather than waiting for the executor to send it
non split row parser shouldn t trim lines because whitespaces are significant
dismiss keyguard for brightness qs tile
changed es version from string to class version
rename rabbithealthindicator bean
fix wrong menu resource usage bug
add safety check
make stream description nullable
delete secret content when deleting by id
fix see tag that pointed to itself
changed method name to addqueries
remove unused code
udf abs throws npe when input arg type is string niklaus xiao via ashutosh chauhan
modify weaver pool to look for classes not only via system
cleaned up configuration javadoc
fix failing test after add test lib jar to test web app
in error messages use the userid instead of the accountaddress property
fix npe in the edit ticket page
oops fix typo in last patch
library roots should not contain module sources
add assumendkisavailable to androidaarintegrationtest
remove glstate checkerror from rendertexture
fix a bug with how edge ids are created
notify the output service at startup whether output tree tracking is enabled
fix javadoc formatting in resourcehttprequesthandler
fix broken merge
count down latch if we don t have any changes in cluster update settings
suppressed warning in test
fix concurrency problem
build fix compile error
fix broken messagingstyle
remove unused import
move networkutils to util package
add category to basicbackpressureservicetest
send end_padding to server
resolved npe when owner is not null
silence excessive logging from scroll events
fix atmospheredotxml location issue
failure in localhistoryactionstest teardown shouldn t bring down other tests
fixes an issue with audio level caching in conference calls
test testcorruptiononnetworklayer had a typo in test name
corrected the b size algorithm
remove logger setlevel fine from javautilloggingsystem
remove backup of sleep policy setting
fix test failures
remove old test code
use defaultexecutionresult instead of custom class
add final to ep_name
remove this line
add a hidden configuration for the muninnpagedfile stripefactor
fix javadocs warnings from the last commit
updated test failure error message
do not load gnustl in nativememorychunk
remove code duplication in graylog2module installinput
remove dead code
fix gouraud shader compilation w multiple lights
attach ideasdk jdkannotations
replaced stringbuilder with string concatenation
remove trailing space
fix small issues
add support for different keystore passwords to the example
camel mina should preserver headers
reducing the contact list should not hide status icons fix
remove unused imports
allow empty action views to avoid the wrapper
remove mediacontroller from videosampleactivity
run the aggregation handler for runtime queries after the runtime stage
add the ability to return doubles and strings from functions
handle resource name not found exception
add origin to failure message
add additional documentation to the delegation implementations to make their intended use more clear
isemptyorspaces accepts charsequence review
remove some sql syntax which threw off eclair
fix a resource reference
remove hardcoded position for scrolltoposition
increase the timeout for tablesizereadertest
allow s not in yields as is symbol
fix scrolling delays
replace commented out code with robolectric logger
must not hold psilock in readaction in tests
don t use ioutils here to avoid dragging it into the tooling api dependencies
add parentedge getter to partial street edge
add wireless power adapter judgement
display command message closing bracket correctly
provide possibility to set user defined origin name on cloning github project
add one more todo
undo pressed animation when losing focus
fix npe for devices with hardware navigation
remove unused method
explicitly specify user for public mode
use existing mapping when merging
fix trace logging statement in zendiscovery
avoid race where request snapshot contains null
remove methods accessing call sounds
don t set unused parameters
add missing implementation annotation that was breaking the build
adds special case in hostnameport matching returning true if no host is specified
call webview performlongclick instead of performlongclick
add override for longrunningoperation setcoloroutput
add the name of the tag to reader_tag_previewed
add voldemortapplicationexception handling inside getcallable getallcallable
do not include stream url in email if web url is not configured
erased commented lines and unused imports
add alpha management
also fix the rmiroutetest
importing all the public sql functions and implicit conversions
allow access to the underlying disklrucache
fix edge case where session counter might go wrong
fix problem with null intent
ignore width and height in img tag
add addurl method to moduleclassloader
set instance status using applicationinfomanager instead of directly
set build fingerprint for the new user
fix crash in the uiactionsheet constructor which takes button titles and
remove debug print
catching all exceptions in gelf handler thread now better safe than sorry
test assert against mincompatversion instead of a static string
remove switch support
remove unused import statement
ninja fix a typo in dropuserstatement
extracted line_separator constant in printer
fix npe if effect is not defined in the xml
fix build break
fix tablayout when used in a horizontalscrollview
remove unused dependencies from helper class
re enable identity transfer vec creation for non enums add helper to make a number sequence
add my testnet dns seed
don t use strict mode here
remove pbuffer initialization warning from lwjgl
do not add libraries in modal state
add todo for consolidating array of services
fix an offset bug when imputing missing categorical values
add getstoragelevelalias used by jsp
disable save_surface_to_pixels_hack and a few other init tweaks
remove unnecessary comment convert to lambda
add ability to return the currently used server addresses
migrate data_activity_timeout_mobile wifi from settings secure to settings global
add nonlocal reference to the inspection
remove unused reference to internal method
ignore a more genric exception
fix broken build
suppress unchecked cast warning
ignore inre from docustomizecellrenderer
make matchwithresult more robust wrt to null strings
terminate worker when failed during initialization
preserve nops as they may be needed for compiler plugins
improve exception text
trim input username and password
fix safe key caching
add minor optimization to abstracterrors
fix bug in setting launchables
moved sequence number field to right after frame length field
mixed up uid and user id
turned arround version comparison for preventing test of a certain level being executed
fix nullpointerexception in mediaplayeractivity
fix pythoninplacebinary rule key calculation
remove debugging output
fix filegroup paths to work with external repositories
do not show renamed library as error in dependencies tab
fix the access denied statement in filechooser
forget the reference to the activity after creating everything
live streaming video may not have the right size at prepared
recoveriescollection findrecoverybyshard should call recoverystatus tryincref before accessing fields
close the urlclassloader when remove an add on
remove extraneous request n and oncompleted calls when unsubscribed
empty message implies no message
add additional unit test for empty tokens block in grammar
remove the atmosphereresource in case something goes wrong
after a monitoringstatus restore beacons that were inside have their initial lastseentime and may trigger quickly as exit enter
fix tag state identification problem
remove singleton annotation
remove the message format option eventually it should just use one that works with maven
dont need second identity keyword in hsql
fix bug with default selection navigation settings
add convenience methods to bucktracing
set getscrollableviewscrollposition to protected
add configurationproperties to autoconfigs
add fragment view injection support
remove final method qualifier to match abstractpartialfieldproperty
handle subsequent stop requests
remove open project in new window from web popup menu
don t normalize eye vector
changed requestparams getentity from package private to public to allow standalone use of requestparams
set state with high priority
calculate the real path of a link s destination when on windows
fix output stream to not use the converter in the buffer but do a simple hard coded conversion
selecting treated root generates invalid hql
tidy up error case on enqueuing
don t quote boolean values by default
hide home on demo activity
throw an exception if the selected folder is not the original folder
make lock fields private
fix bufferutils on android when used on non bytebuffers
add dnsseed bluematt me to the dns discovery list
disable wadl generation for jersey
change collectiontype renderloggablestring to return uninitialized when collection is uninitialized
remove reduntant line
smallfix remove explicit type arguments in storagedirview
revert updated distribution download urls
add developer version flag
fix npe in mainmenuactivity
fix broken int test
add a javadoc note to indicate that this method will be remove in a
only fire the callback for the imageview for which the callback was provided
fix npe while using mlayerpaint
fix samples in dsl docs
add more javadoc for episode release time and next episode
restore continueonerrorforinsert to restore buildability of code
remove some comments in unit test
remove useless code
set last feedback time for upgrading users
change default constructor
implemented onbackpressed on shadowactivity
add setcontentlength method to cacheresult
reduce the log
add common attribute ignorenulls
properly handle return statements without expressions e g return
convert groovy conf path to system dependent
revert don t use static viewholder
add nativeservices intialization for forkinggradleexecutor
check write access in setproperty
remove unnecessary cast
call timer record with a timeunit
throw exception when parsing invalid xml
changed texture wrap enum
fix a bug in the getopposite method in dhns
created minimal test for profile storage on off command
fix type in twitterrestclient
fix the build
add detection for edit mode in rendersurfaceview
cell changes should also flush column precomputes
add constants for true and false to pactboolean
updated argument types for urlmappingkey from collection to set
add license header
fix accuarcy value in dependencyparseritest
remove outdated warning about rest api streaming
turn off perf debugging
remove check that executionresourcesexhausted call must be from environment thread to enable chained tasks
fix compilation error
remove unsettable crosshatch parameters
eliminate production of useless anonymous inner class by javac
optimized showbaserevisionaction update
align the structure of hasheq dohasheq with equiv pcequiv to avoid deoptimization of hash by hotspot
add fall back package id for voice recorder plugin
add missing super calls
core sample add todos
stopped the not editable userdata warning to show up
add check for npe
fix localejbreceiver to pick the sessionid from the right place
remove dead code
modify the definition mistake of license header
fix unit test add null check
remove unused imports
prevent recursive loop when reconnecting
silence a findbugs warning
fix unit test
refactored tests for quantified predicates out into their own test case
remove unnecessary cast
vcs log show root name in tooltip only in multi root case
depend on interface instead of implementation class
fix bug on collections in sql queries reported by cgandelli gmail com in ml
add note about getmultiuserchat in muc s javadoc
fix objectchronology constructor to use converter properly
fix slight turn definition and remove redundant lines
fix style issues in inputmethodinfo java
put contextstorageoverride in io grpc override
save input method subtype even if the current ime is not running
directchannelbufferfactory getbuffer bytebuffer must not return a read only buffer
remove the unused buildrulefactoryparams parsevisibilitytarget method
change syslog processor log levels
hotfix remove leftover config key constant from executionconfig
fix missed id in python test configuration converter
removes outdated comments
add setbindings helper functions
remove duplicated code
don t get the servlet context till after the deployment is started
fix small refresh problem
fix problem with running already finished actions by parallelaction
fix bugs wtih js and card aggs
resourcebundlemessagesource checks containskey before calling getstring
fix another off by one in a comparison
revert revert preferenceactivity now applies its theme correctly
add configuration command class
add spacing to banlistcommand output
fix npe retrieving the user s roster
wrong condition when evaluating if to fix null focused component
remove unnecessary wrapping into fairdiffiterable
render top back and back button on search results
replace mention of bobs in class javadoc with spvblockchain
switch statement found where one case falls through to the next case
fix squished ui elements
create interfaces for extensions with static methods by convention
use belongto to compare the current location and the oldlocation
fredricho fix multiple fragments in stats lists on tablet
replace calls to deprecated httptransport send
remove unneeded static modifier
changed some of the language in documentation
remove unnecessary initialization
modified response java to improve debuggability
fix compile error add clobstat
fix symmetricencryptionconfig npe
more items specifies wrong value
fix test expectations
ignore mediascannerconnection leak
remove obsolete comments
fix charset decoder to return empty string instead of nil
rest cell value command enablement fix
improve exception message
remove unused variable
fix test for editing
prints some extra info when dropping stray invites
remove unused variable
fix default values for debug and signal strength logging
record type of whenclause in expressionanalyzer
can t find com sun xml internal messaging saaj soap localstrings bundle
add pydocstringtest to all tests suite
initialize mcolorwheelrectangle in onmeasure not ondraw
clarify use of system nanotime for measuring eviction times
remove welcome message from testdiscussionhistory since the server no longer sends that message
tablesample percent uses int to store the intermediate size need to use long instead
auto dismiss lockscreen when switching to guest
replaced usage of ivy message in recently copied code with logger
remove unused blavatar
add license header
avoid npe for camera with no params
exclude nulls first and nulls last tests for cubrid
updating faunus incr
add player get setlistname to change the name of the player on the player list
suppress unchecked warning in generated code
remove the duplicate string literal inspection
debug available columns selection in data laboratory
remove setuseragentstring from webviews that display external content
add stringlist object constructor
fix build error
fix travis error on an empty vehiclestr
update foo endpoints
output the total graph build duration
fix a bug where calendar would sometimes draw incorrectly
use immutableset instead of unmodifiableset
made a singleton out of the stringandpartitionawarepartitioningstrategy
remove import statement
remove unnecessary code
remove all double single quotes
add a todo
fix last test case
catching only command failure exceptions when getting connection id and failing
loozing exception handling
disable failing test in hdfs ufs mode
fix typo in javadocs
use other test port
fix method name in javadoc
debug log on testmergeshappening
add missing timeline test for versionedintervaltimelinetest
add null check
fix failing integration test
use the default dummy identifier instead of xxx in smart completion
remove soft keyboard while entering text
supported unique indexes
add some comments on auxiliary trees to tsurgeon
remove check for missing parent calls
remove invalid dependency
changed linkedlist to arraylist for random access
remove unused code
remove unnecessary imports
fix possible npe in snapshot service if a shard doesn t have primary
allow apps with score_networks to set the active scorer
reflow long comments
fix regression in propertysource placeholder resolution
remove unneeded calls to string contains
fix up typos in documentation in instanceregistry
add missing semicolon
fix issue on contains operator
refined default excluded packages
don t log error exceptions during gdsl processing only info
move action bar and view init to separate methods
remove unused field
fix broken preconditions
replace strong interning with weak interning for better gc
fix bug causing npe if non whitelisted player joins an fml server without fml installed
add schemav3 to the list of classes that need extra help when discovering all schemas
adds missing bluetooth as network type
velocity freemarker tilesviewresolver only return a view if the target resource exists now
add a hidden method to clear the capabilities of a networkrequest
hivemetastoreclient should close existing thrift connection before its reconnect chaoyu tang via szehon
hide our shame from the javadocs
in order to unpack a war remove the directory before reload a war
fix cast exception with selectable text
add extra catch
ignore tag html in parse tree as created by doc
remove unnecessary log output
consistently use varargs signature
report the new size to wallpapers when the requested size changed
fix problem with update rules being executed twice
add notes about weld and container discovery
add marker for minor clean up issue
render only one method separator if there are several of them for the same offset
don t warn about uninstrumented var set as it doesn t block for real
add a fixme
remove a sort on edge list
java made method static to avoid this need
fix live wallpaper touch events on older api
change details action name
set batchentryconsumer sequence for advancement directly from the entry to prevent getting out of step
add rstudioapi to shiny gadget dependencies
add initialization for mysql driver in server module
fix bug unsharing in maintenance mode removes the file
fix mojo integration test to use system property or default login url
fix some quality flaws
add explanation for tvview
stabilize startup of es in tests
do not start scheduled pings until transport start
fix closing state
log test run errors to logger
remove check for consistency
fix compilation error in externalextensionsdialog
fix merge issue
add singleton scope to navigator
set initial text for edittextpreferencewithvalidation dialog
remove runwith from hasotherannotations
add comment about why stringbuilder is used here
deprecate unused protected field
fix bad resource label
remove todo and improve javadoc
log replaced methods
npe in got log do not set icon in search field for mac
add value null check
prevent drag n drop nodes to themselves
remove useless code
fix wrong parameter in hashmap remove
cancel inaccurate fps performance pre measure
obtain a lock while iterating over the synchronized set
restore old defaults in tests
add tostring to systemlogrecord
remove expired cookies thanks neromancer
fix npe in planprinter
add shadowview requestlayout
fix issue with symbols vs attribute name disambiguation
fix bug in binary protocol
sourceformatter add languageutil format
hotfix tests fix manual tests to properly start testing cluster
fix laser data not existing in some cases
modified rate statistics to just return number not a rate
if new otr password set remove the existing instance
fix a couple of npe caused by user tapping on a chart which has no data
add thread dumpstack to warning from shadowactivity findviewbyid when contentview is null
fix crash when making emergency calls
fix javadoc comment link
fix bug in task manager
make test suite mostly pass by avoiding closedchannelexception
fix small issues
fix cubeimageview display last viewholders image when imageurl is null
hide old toast immediately if a ne one is requested
changed system exit to runtimeexception
end the animator when stop is called
change the default type of the page recycler to concurrent instead of soft_concurrent
remove tweaking of auto commit in abstractroutine
add todo for future work w r t presence of chat room members
fix reconnect reassociate handling
fix bug on record tojson
remove testfirstandlastleg method
remove an obsolete code comment
remove unused imports
add shader settexture call in pshape3d renderfill
fix edge case where pixmappacker eats all memory in infinite recursion
fix oom errors that happen occasionally
updated log from debug to trace due to excessive output
remove tests for offline messages which failed the build
toolingapi for idea add missing serializable aspect problem exposed thanks to the samples test
delete some unused stuff
simplify updateassignments in clustermemorymanager
eliminate eclipse warnings
remove import of backport util concurrent arrays class
handle missing fragment crash
add onreset to batterylistener
handle editor reopening errors
rename the dismiss to dismisswithanimation
add corners to circular crop window
fix another test
make assertions stronger
add a method to get the recipe
set active to true on indexing ops
corrected javadoc for remotesshprocess add remote local tunnel
add message sendtotarget
fix cut paste typo
improved logging statements
fix rlock s entries
remove ugly underscore
use relative key to get relative timerange info
do not create outgoing connection if hostname is empty or null
comment out the xz compression as i got an unexpected end of stream error in amazon emr with
resolved some checkstyle issues
improve javadoc comments
fix time conversion issue with expiration policies
allow wake keys to wake dozing device
fix formatting in usage message in appcompiler
fix cs error
change kryo link from google code to github
fix bug on reload of document in query
switch right bracket to left bracket on condition start tag
fix sync issue which could cause multiple disco items entries
add abstractprocessingfilter getauthenticationdetailssource
add processing opengl
namepsace the udf context key to avoid accidental overwrites in the future
fix failing test for duplicate variable inserts
fix compilation problem with generics
add missing word forms
remove unused code from slb
add null check when header is empty
two fix typos in java source files
fix camel casing of config parameter to be the same as pom based parameter
encode success data
remove bogus to treat jsreferenceexpression as psinamedelement
fix nasty bug on lazy unmarshalling of collections
fix realtime integration test
add comment on candisconnectandkeepplaying
add override to cloneforspatial methods in examples
revert invalidating a te should not cause the pipe to drop
improve compatibility with su implementations
reintroducing deprecated addconstructorarg variant
improved rb tree algorithm for a better balances tree
fix tostring in location to be readable
even on failure don t return a body for head request
static methods in interfaces should not have abstract by default
add rate option parsing to graphhandler
set tests before push task to stepik
fix comet handling in case begin event calls cometevent close
add warning about randomly generated keys to wallet constructor
improve some comments
make the protobufparser timer daemon and labelled
skip the testmetricusage test on windows
fix typos in fieldsortbuilder
add one more unit test
prohibit in fqnameunsafe
remove lies from a comment
displays the exception simple class name for unrecognized exceptions
allow yuv_420_888 as a yuv type
set ultimaterecyclerviewviewholder as public class
for community server use the community launcher
use guava s iterators class and emptyiterator
fix a bug where channelfuture setfailure null doesn t fail
give each thread its own parser
don t call shutdown to avoid rejectedexecutionexception
add getter for cullingarea in group
always set pwd variable to the value of working dir
add pause hook to shadowfragment
optimization keep info if user is anonymous
do not log the interruptedexception stacktrace
add command line constraint
remove the unused import of rabbitmqconsumer
also backing out the test
remove unneeded local fanout for semijoin
make converttohex public
add an escape route for cases where a gate is on all sides of a pipe
master should create the config table if it s not there
fix freetypefontgenerator had wrong encoding
catch throwable instead of exception in driver execute zhihai xu via jimmy xiang
add missing semicolon
make noopprojector instance final
skip hardlink tests under windows because it fails on new agents
fix npe in hgerrorhandler
fix merge errors
ensure headers committed immediately for sseemitter
fix an initialization issue in motiontrack
remove callback when the job is done
add convenient changetimezone method
remove unused code
trim the value before add to the list
simplify dead code
fix seekableviewsfortest to handle the reset long double overload properly
add comment explaining testenum tostring override
always fetch the dir s blockinfo remotely in getclientblockinfo
fix pycharm community detection
add class to error
fix qs not collapsing after some qqs gestures
remove unused parameter and never thrown exception
ignored examples java tests since they re not regression tests
generate default policy for devices without imsi
mute the connection reset by peer exception it s spamilicious for the most part
generated forms for faces elements
smallfix remove redundant empty line
fix test failure in testrestliresourcemodels
generalize the javadocs on cachebuilder ticker
add httponly to cookieretrievingcookiegenerator
fix hidpi border for commitinfopopup
add reference to expectedexception
remove return statements if there are no side effects only
remove dead allocation of unused thunk
add size warning to textureatlas
remove outdated todo for localelist
fix javadoc of graphics setcursor
fix honeycomb crash
do not display the ime when selection in non editable text
add logging for output disconnection and reconnection
add a default constructor that uses sensible internal strategies
add check to prevent empty names from being add to the basenames collection
remove warning in glmprogresspage2
implement isscreenbrightnessboosted in bridgepowermanager
fix assertion in broken test
remove unnecessary import warnings
add gl_patches to convertelementmode
fix fastimage sets color of sprite as well now
add new attribute to ocfile to know if this file is shared with users groups
don t have the constructor private
only add property to relationfactory if
removes debug printing
fix simplequerystringtests testsimplequerystring assumption that depends on how documents are routed
intercept empty feed early on
add external system rt jar to the classpath of the remote process
restored the static logger back
add missing charset check for file
remove unused imports
modulemetadataprocessor mutates the meta data
use the darker gray color as a default for the disabled state
remove extra trace
add clarifying comments around javaresolveutil getcontextclass
revert action state fix
change resourceshrinker instance of charset defaultcharset being used to parse resource file to use utf_8 explicitly
remove unused constant
refactored out a static method
pass all exceptions back
changed the size calculation expression to make it more readable
git place testroot into a subfolder of the temp directory
fix issue with ordering of checks
add system api for disabling touch events in wallpaper behind the window
add training samples to progress printout
remove interface disable in wifistatemachine
upping max deviation percent
add method for getting a place by id
remove unused compiler arg
turn off attempted fixes
increases visibility of workflowid ctor adds getters
only process a single file when performing hotswap for goog defineclass
fix description message after successful installation of package management tools
cannot change combobox values on wrapping and braces page on mac os
make unnecessary exports in goog modules an error
disable multithreaded indexing
add new reserved words to the type system
add null checking in programs genres encode decode
avoid crash not found resource id
switch the default verifier to browsercompathostnameverifier
add controlleractivity onsaveinstancestate
slightly relaxed non negative test
add documentation to websocket classes
adds color to refinery fluids
changed the layout of the missing mods gui a little
don t build eclipse env for themes
fix failing test
don t log the full transaction
fix gmail crash
add client id field to project user object for cursor tracking
fix npe in diff
add getcurrenttime to timetools
give a rest to cpu
add documentation to post playback processing
hide keyboard when message is displayed not only after interaction pending intent
fix assertion in gravesbidirectionallstmtest java
fix dot writer from breaking when there are missing bindings
suggest jdk path on linux
add convenience method tiledsprite getcurrenttileindex
use a single thread to dispatch messages
include elevators when finding islands to prune
commit documents in tests
fix flicker when docking task from recents
fix typo in javadoc
add additional documentation
updated to next snapshot version
determine compositebytebuf implementation by using bytebufallocator
revert unfinished changes
fix broken ut
fix unit test
use selectable to set focusable on root pref view
fix wrong license
remove unused method
persist widget specific config options
fix bug in previous re fact
add done for cookies in the feature list
really fix npe
moved javadoc from value to monad
replace extends of deprecated activity
specify size in toarray
polished the cxfendpointutilswithspringtest
deprecate unused code in o a catalina deploy
remove unused import
fix processing of persistencecontext on fields to use the correct enc name for the binding
fix quality flaws
fix the build
updated cordovawebview to experiment with onscrollchanged messages
remove value check in getrecordornull add istombstone to record
smallfix simplified equals implementation of getstatusoptions
remove redundant constructor
rebuild xcode symlink trees on buck version change
fix logging level of quorum warning
fix resources operawresource int typedvalue in layoutlib
we use fnv hash functions to generate a large indexable sequence of non colliding keys
fix incorrect syntax
add a configuration parameter for servlet context init parameter use
fix checkstyle issue in blockidtests
fix grey yellow issue
make goto declaration work in groovy lookups
remove repeat annotation in joinstresstest
i can t gen dbdoc from changelog like this
ui perform action instead of showing popup if canbeperformed for action groups
modify equal test
registered the wrong high id was object id but is now hash position
uses recordload normal for natural errors on
fix fetching accounts by jids
remove getlocalname from reference
fix a possible deadlock when receiver thread invokes mapmemberadded while ping thread invokes memberalive
fix duplicate resetdeltarequest call
don t create new instances of filechannel
task put returns success true
do not preload actions in unit tests
get rid of stupid compiler warning
do not create temp file to show diff for binary file
add class description
add iscancelled method to safeasynctask
detect pdf js window closure on mac desktop
hide captions when the track is disabled
remove import that should not be here
add missing osgi and usage modules in tests
update measure to measuredeep
add a custom tostring implementation
log lessen logging because it may happen e g for origin head
throw unsupportedoperationexception when creating ephemeral
add ability to remove stat
pulled error message because its a conflict for convolutional nets structure
remove some dead code
re use earlier retrieved internationalization string
reset mime type after custom mime type test
maybe we need a test for this test helper use correct message in assertequals string version version
add missing tool tip resources
properly init the lengthsquared and some small reformats
deprecated uncompleted classes
made the vertical scrollbar to be displayed when needed
don t bother pointless attempt to construct thread pool with no open circuits
consolidate bsd targets test
do not add duplicates to mwaitingvisibleactivities
reducing number of trials for travis
only overrides back button when sheet is showing
andres yann don t place create app button to the right when creating a grails application on mac
simplify atnconfigset toarray
corrected remove space
improved message on error while sending records to the client
change the usage of settype to enable changes to isdynamictypedfalg through the constructor
fix missing import
fix build from variable rename
fix compile error
remove unnecessary suppresswarnings
do not use modality state on pooled thread
don t call checkclass for lightweight edges
made npe in httpresponse debuggable
expose the highlighter builder field constructor
turning off global jar caching to figure out why it doesnt work with classworlds
remove rect allocation in each layout pass
do not add parent to null row keys
add dependency on tests if necessary
add org objectweb asm
changed default stack size for running ant
fix up merge
improve the glm admission error message
self presence status element lacked code attribute
restore api for plugins
fix complex character support for plugins
identity combine other should just return other
add querystringencoder tostring
adds new convenience constructor for initializing worlds
expose the current transport in the request attribute
properly send view for manage lists dialog add description
give the file path in the exception so that it s easier to debug
use interface rather than standard implementation
fix memory leak
if sub server cannot be bound log it as warning
add meta refresh and location to outgoing links
fix missing capture result callback issue
improved error handling in mediaplayeractivity
remove unused method
remove superfluous imports
fix default value for share contacts
improved exception message for the case that a registering
fix bad defaulsubscriptionmode spelling
catch all exceptions when parsing or doing network stuff with thetvdb
downgrade gen aidl message from error to warning
remove debug code
weaver needs classpath inherited from parent classloader
search importspec in stubs instead of import keyword
only set park and ride or kiss and ride flags when transit is enabled
renaming finding attributes tag tag file function from tld
calling correct method to calculate offline segment document count
revert change to replyexception message but provide descriptive tostring method
make exception for creator
use the correct number formatter
fix flash of uninitialized surface
remove unchecked warning
fix import style in moco file root test
add a name for the folder observer thread
connection settigns aave fix
addworkerhook shouldn t accpet null and add test
restore deprecated api
dont keep instanceof state and use correct coding style
fix log format
add reverse iteration from tail of pooledlinkedlist
remove synchronized from linkedhashsetwrapper add
add missing javadoc comment
adjusting lock level
tweaked java sample error printing
add javadoc for showcaseviews addanimatedgesturetoview
add back missing imports
add player get setexperience totalexperience and level
add some javadocs
include server name in plot mru for symmetry with doc mru
support comparing the builtins namespace
remove duplicate condition
add presenter and stream for user status
improve javadoc for lazy loading example
kill unused logging logic
fix crash in static drawer when no indicator bitmap is set
add an example of a javascript prune evaluator
use null watchservice as loop terminator for pathwatcher
restore goto line number in goto file
tapping outside the system panel shouldn t open the recent panel
add missing copyright header
fix typo in exception
fix checkstyle issues
improve error message for column not found
fix aioob when vector is smaller that ordinal
avoid stream leak upon exception
fix welcome files typo
fix overrides of islayoutrtl drawable dr
run lines should put focus in console when selection is active
commented call to removeallchildren from non awt thread
fix random stats generation for debugging
add trace to hunt down stalled build issue
remove unused code
lets keep things as simple as possible
add an isempty helper
stop unit test quicker due graceful shutdown wanting to be too gentle
automatically create sdk instance for directory based project if none exists
restore and deprecate api for a while
tests fix avroexternaljarprogramitcase logging
setting positions for parameters even when running in nb editor to avoid problems in javac s flow
read the element count of a packed switch payload as a ushort
fix test on cache
change the property name about idp from telephony
give removal dates releases for hashfunction hashstring cs and
make the window have a somewhat sensible title
avoid top hit providers duplication in se
remove symbols in lists for setting groups
clowns may check their sense of humor somewhere else
adds simple refactoring to the createinvite method
remove encoding argument from source it doesn t work anyway
fix category call intent support data
fix recents on crespo
android use setserviceuuid to build scan filter
add null check to copy selection
fix intdef annotation javadoc
clarify what fancy means
fix version migration of nested executions
add a final qualifier
change menu_refresh_id to be different than menu_sync_id for the code to run through different paths
provide applicationcontext instead of application to init filedownloader
fix test on windows
add javadoc to the class and remove an unused constructor argument
optimized getmaximumvalueforset method
rename class to buildoperationinternal for consistency
fix documentation copy pasta for gettranslationy
do fast checks first
changes set behaviour to old one before changing logic of zero ttl
fix a bug where the media template wasn t transforming right
add name to thread used by notificationservice
rearrange screen methods to represent lifecycle better
remove partial getelevationprofile from streetedge
change the log level as this get quite annoying when another framework is using it
itemstack no longer implements serializable
don t perform dns lookup in androiddotnet proxy islocalhost
fix beatrix test failure becayse of chnage in payment
fix typo in rosterentries
fixes a windows path issue
automatically start and stop animationdrawables
add some javadoc
stop databaserulemanager in stopbeingleader
maintain private chat on muc
add getbukkitversion to server
android bump version
clarified javadoc for scrolldown in solo and scroller
remove isreference from symboltype
remove isolated use of reactor buffer
remove potentially misleading exception message
add boilerplate for jetty security work jesse is doing
remove unnecessary method
swapped yaw pitch in location constructor to match minecraft
resized image for single shot activity
plugin remove deletes bin directory
debug info remove
use noncanonical paths aware fileutil getparent
do not notify start if console is already disposed
fix jre self cycles
improved generated action class tostring
fix broken test
give all_privileges when create a share with user group
don t clear notifications on login
enable cleaner hack so it consumes less disk space
send empty text with statistics message to avoid producing unnecessary output by idea project runner in teamcity
rename exported content pack filename
fix path for new layout
postgre struct types fetch fix
save tile to correct place
explicitly add rt jar to the script class path
fix compile error
deprecate tvtag show id column
idempotentrepository add clear operation align ftpconsumeridempotentreftest with clear operation
revmoe failed testcase
add additional break statemement to please findbugs
remove final declaration of onpreexecute on onpostexecute in order to further extend methods
better debug logging of where the default mapping is loaded from
fix revlog where start of range is the zeroid
tweak support action bar indeterminate progress bar display to mirror that of the native version
fix default missing model zoom text not being set to false on all places
return a copy of the downloads
add xpack as official plugin
fix joins that have order by rand
remove one the checks i add in earclippingtriangulator seems to break things in certain cases
improving msging when indexing service is not found
do not convert object names
don t display negative unread count in message list
include paragraph tag when separating note from documentation
set copywebinf to false by default
add null pointer check for load balancer
re add missing blocks to post processing
fix the build
skip test which fails occationally on ci
allow any exitcodegenerator bean to switch of auto configured bean
remove an unused constructor of inputmethodsettings
compensate for eclipse vs sun compiler discrepancy
take scanning files for index outside of read action in concurrent mode
add one pixel splitter
fix completion of elements outside the file prop files imported files etc
reading definition from file as an alternative to system in
preserve fonts subpixel antialiasing when popups fade retina get rid of commented line
manually delete cachefile before we write to it
have one read only and one read write fileresourceset tests rather than two read only
fix some tests
add another fallback case for androidmanifest
sendbroadcast needs protection
add unit test to claus camel jaxb fix
save activity reference for welcome dialog
add parsing of cython typedefs
remove duplicate log
fix multiuserchat enter nickname check
fix bug in profiling which occurred together with lazy task deployment
close counts store when dump is complete
add javadoc to configurationsource
remove duplicate since tags in embeddeddatabasebuilder
never expose user dir to the web on directory listing
setting selection state when opening a note
hide popup on detach
fix npe when loading a graph with giant component filter
clarified javadoc for searchtogglebutton string int in solo and searcher
add some npe protection
revert add tripid to legs
do not log pce
disable action if not applicable
java code style help topic id
remove test codes
do not compile java part of runtime
use the correct file
suppress all filesystems here due to jimfs brokenness
fix broken link
remove a useless assertionexception
add ast_file_lookup to the list of key types whose skyfunction implicitly uses the pathpackagelocator
fix warnings and style
add a possible todo
fix bug of throwing error when saving model with none field
o add license header
move the creation of readeruser to the correct place
knocking out warnings
fix tests related to material autoupdate checks
display home as up on uploadactivity
replaces metal look feel
add println when all the tests pass
changed immediate button as neutral button
add internationalization support for content disposition header of response
fix incorrect id
throw parsing error if match phrase query contains multiple fields in its short version
set willnotdraw false to draw without background set
fix systemstore constants and got rid of the superfluous zone rep
use all available changes to detect operation name in revertdialog
kept the api in genericfile
add x requested with
deprecate unused code
set the default rendered to not be opengl patch from ben
add missing whitebalance mapping
reduced steps of search by always leaving the last entrypoint
set context class loader when creating connectors
fix fluid blocks
add command line configuration option access method for filtering desired properties
fix listener name blankjavalistener javabaselistener
add test to resultsetiteratortest
makes closurerewritemodule respect suppressions
make pyunbound local variable inspection enabled by default
fix bug when restarting asynctask
revert milestone builds can t be a most recent release in cross version tests
fix last expandable item in expandall
don t send unneeded messages
only use the context xml from an unpacked directory for a war if unpackwars is true
fix incorrect usage of assert notnull
enum s valueof already throws iae no need to do it ourselfs
add addimport for entities
remove unnecessary null checking
add the original copyright
remove unused import for now
add thread safety annotations for tachyon resource in common module
add check if error not in ignorederror list
expanded a compact statement in ephemeralfilesystemabstraction to help track down a flaky npe
fix disposed error on run cell
make threshold final
fix bug where loadermanager instances were not being remove
fix starvation in backgroundhivesplitloader
copying consistencylevel from thriftcolumnfamilyqueryimpl when constructing abstractthriftcqlquery
fix break on calculatescore when dataset null
permit cc objc deps
add support for bsonobjects in output writing
add show hidden files toggle
fix dispose for custom disposables
fix typo and javadocs
fix popmatrix condition in pshape3d draw
fix pubsub affiliation getelementname
enhanced zaprelease tests
fix another bug with rtr to remove things correctly from running tasks
fix the test in perforcechecker remove findbugs warning
add min max temperature in one string
clientsidesessionmodule remove println
the error level shouldn t effect compiler output
add a try catch to fix a background refresh crash
fix a platform specific regex in abstractpollingreportertest
fix several bugs in addfeedactivity
setting list divider programmatically
remove dead code
vcs log optimize imports
log set bek synchronously when applying filters to avoid locks
add friendlier configurationexception for malformed abstracttypes
remove unused import
remove nocommits fix test case
fix clear interaction count for new quotes
add set getconf to forwardingfilesystem
allow white space in the property name
use one thread to process rtp packets
add support for compressed http response
fix the cs errors of camel spring and camel test blueprint
remove unused imports
remove vertxgen on pumpimpl
remove add todos
investigate testbeelinewithargs jason dere via ashutosh chauhan
matched up span exclusion types
reduces a few equals calls
make method protected to prevent accidental usage
add dbmaker snapshotdisable option
fix a crash when deleting list from list settings page
add timetable snapshot to routing context
display log on shutting down only in ismonitored mode
pce from running invalidation tasks should reschedule file processing
add getusernames method to stubzmessagingapistore
fix client login issue when logging into a non existent dimension
add livingentity haslineofsight
fix uri to have this test running
threadpool now uses daemon threads in cudacontext
improve javadoc formatting
put emergency first
fix missing fields in cli show schema output
fix free downloads
fix seek bar start view
make getter s comment simpler
fix quality flaws
remove a duplicate codepath for folder insert
only show dialog when really out of memory
do not spawn gradle daemons during test execution
add clientid related methods
clean up new embeddeddraweractivity
do not ignore paths without home variable if home is not defined
trim fetched show titles
remove unused import
apply walk and bike speeds
remove unused enum
remove weird matcher abuse
fix show release time check
remove redundant check
closing the connection if frames that carry headers don t have the
don t call field get
fix xcode project generation always generating all projects and workspaces
fine tune lead time
add back remove method
hide the soft keyboard when pausing the activity
add disable async saving hint
fix broken travel time calculation hops vs stops
tweaked gtasks auth timeout and logging
make talkback or other accessibility services updateable from playstore
improve javadoc for ontune and notifyvideoavailable
fix failing test
resource value class member was using the wrong name streams vs outputs breaking the output listing for a stream
add method to algorithmoptions
fix no action button with resource string bug
remove debug text from title
remove getter functions from subplan
add serializable to defaultoauth2clientcontext
don t freeze the screen when resizing task or stack
improved error message thanks to paul roebuck for complaining about the old one
remove a always true if condition
remove myname accidentally included in one of the files
moving exception handling around a bit
changed newinstance to protected in json
add override annotations
add java version check to zincscalacomiler
add yahoo and lowercase v for legal cases
fix unboxing warning
add index name to the upgrade exception
demote warn log to info
fix bug on unload of index
smallfix incorporated pr comment
remove some logging
fix map extension recipie
convert mappingentry to kotlin
delete some whitespace
fix last minute problem with new ingesttest
fix code example in javadoc
temporarily disables id buffering
add entitybinding getattributebindingclosurespan and getattributebindingclosure
remove unnecessary override
fix javadoc for leanback
correctly show watched button after add movie
adds a forgotten graphics dispose
remove invalid test
fix another bunch of vert frag typos
fix a bogus comment in checkmerklebranch
create error elements info using base language s elements
add corner cases to equals method for parameter description
seriously we didn t deprecate this for a year
made playerevent breakspeed fire when a player cannot harvest the block
fix mis spelled flag name in error message
remove unused providers
remove unnecessary newlines
fix code to compile
override equals in blockinfoentry
add null handling in get invoice
clears db directory
android update wxsdkinstance java modify np
fix button size for guimodlist
fix the gwt build
allow multiple calls to txmaker close
add a log and a comment
hotfix checkpointing fix error message in savepointloader
add more trace logs for debugging
fix copy constructor
remove unused import
test the overdue notification queue
restore wrapped resultsets
remove some unused window manager methods
remove unused and ignored test
fix bug with invalid directories for mac
fix endless error
fix race condition when fetching list of jobs
fix null pointer issue
check errors after init true
cleanup remove unused imports
add the connection s hashcode to tostring
updated to ensure future uses of commandlinehelper use the correct system in out
do not generate docs for self param in class methods
ignore an illegalstateexception from the ping timer that can be thrown occasionally during a shutdown race
add missing param in resourcemanager set inflateexception source
jaxb scan skips annotated interfaces
remove comment characters
add method to listener to notify when step is reached
increase the setup sdp delay to cover for all the devices
deleted commented line
support quoted html color codes
fix unfortunate typo in affinetransform
set undo manager for ace
use a file type instead to obtain psi if language dialect is not set
remove unneeded nosuchbuildtargetexception from throw specs
avoid error on malformed aztec code
allow any character as first letter in package names
moved validatenumerics last in validation and is not throwing any
undo spencer s one line change which had a strange overlay side effect
fix the regressions introduced by the support for dynamically installing bundles
add dun to list of supported apn types
add missing license header
need a codegen wrapper here
fix bug where tapping the comment context menu item in viewpostsfragment didn t do anything
add new constructor to populate a odocument with a map object object
add more data to serialization when tov or formv are nulls
if persistent map is corrupted then delete the file
when using workers for java compilation build deploy jars and source jars on the local machine too
changed deprecated for deprecated
log eip log an error by evaluating simple language expression add unit test to show the issue
moved contentupdate receiver in try block
remove debug comments
add alternative method to create archive fix typo
add proper tooltip for install uninstall buttons
display track distance in save track option in settings
workaround to hide the to attribute with the fake jid
reinstate escape hatch for findresource
disable exploded mode for test execution for the moment
make annotations bg brighter
include output path in rule key for cxxlink build rule
add a null pointer check to make the static analysis component of eclipse happy
catched throwable instead of exception
fix rename in flickr sample
remove the gap before the memory indicator
add name ext to file language
don t run full backups on stopped packages
add annotations for table and column
explain why the sqlcipher libs are loaded at app start
exposing cache stats through producer docs updated
remove getdialog which causes crash when used as fragment
refactored one of the remaining constants
enables partition events for partitions that are promoted from backup as discussed in
make mode local
fix the terrain normal calc for the edges
hide glass pane when search is disabled
remove unused vfsutil virtualtoifile
fix ring volume issue by addressing focus regression in keyguard
fix unit test
hide column header
avoid npe in rescopeglobalsymbols
add isscrolledtotop for recycler
fix references in from import in doctests
temporarily remove test to get build to pass
don t send dead notification before a node completes configuration
disable curator failover test
do not load snappy implementation if its usage is explicitly disabled via system property
add comment split test
add a utility method that will make it easier to get the stringmanager for a given class
saw exception with concurrent modification
add comment todo
reverted last patch by luca molino since breaks test cases
rename verifyairplanemode to setandverifyairplanemode
remove extraneous servers even when not talking to the primary
fix headersmessage parselite
made lifetime available to inheritors of coremodulemanager
always return the total amount of matches in getnumberofmatches
add check for synchronization
change the default ssid from googleguest to the opennet for the power test
make private stuff private
fix a small typo in javadocs
better handling resolve find usages of b field in a b like qualified expressions
populate rangeinfo correctly when acquiring from pool
reorder sections in javadoc
status handler throw svnkit exception if wc is locked correct exception about lock busy
fix null scope in es6handledefaultparameters
fix window sticking around it doesn t anymore
update open files modification tracker on opening file
fix crash on search with multiple accounts set up
hlen calls hdel
add deprecated annotation
marking onretaincustomnonconfigurationinstance as final since internally used by mosby new snapshot available
add comment and made variable final
fix error messages for user using getlogmessage
fix error message typo
remove check for myfile isphysical in texteditorbackgroundhighlighter to restore override marks in compiled files
minor remove my prefix from stepic id serialization name
add ability to set custom print object
fix nullpointerexception issue
add explicit hystrixplugins reset to hystrixtimerthread init of each test
use only move effect for tab drags
progress title should have title capitalization
csv leniently accepts bad lines also when skipping fields
fix illegalreferencecountexception caused by httpclientcodec upgradefrom
fix set argument type of java nio path in basecmd
error class descriptor must not fail an attempt to create a type with arguments from it
add author info
only set transform if it s not null
add null check to point to the origin within the stack trace
add grant flag support to install create verb
add documentation for use of delegatetofilesystem interface
update dox for fill_and_stroke style
fix failing test
avoid double modulation of color alpha
include invalid main resource location in error message
remove oldschool comment
fix small bug
integrate ven s change
switch result bytes variable to long
set extra properties on initialdircontextfactory and corrected group search filter
make suppliers compose function supplier
improved incorrect javadoc of locationindex
fix a bug in the checker
remove ugly whitespace
caught a bug in the stabbing query of the intervaltree
fix concurrency bug with client pool
remove println geometry if fromv tov is null
commented the repository example code
fix possible delay when tls is not supported by the server
removes unused instance variables
add invariant in comment
add default content type header
fix missing import in javadoc test case
specify that websettings
changed the bruteforcepanel to not wait forever for the threads to terminate like done in the scanpanel
covariant equals check compilable ut inputs
async refreshes may post just one afterrefreshfinish event
changed a shitload
allow some leeway here
working on authentication context strategy
stripe button hide side
fix zkroot config key error
apply force hiding of window only when it is main display
refined startup behaviour
remove incorrect assertion
get keep alive rework
remove redundant split generation in jdbc format after fix unsplittabel input behavior
multicast joiner should send splitbrainjoinmessage when searching for other clusters
template not found error
fix type on cast
comment out new whitespace test caused false alarms
delete extra spaces
corrected a typo in the test
include dir listing when check index fails
fix a tiny bug in alias handling
remove the window whose client process has died or become zombie
skip local inspections for big files
addressed review comments
remove the index columns from the blocked provider
nio endpoint should not be swallowing virtual machine errors
remove unused logs resolve todo
prepare osmandmaptileview now use nightmode setting
fix up codestyle of prior contribution to ensure build works
add example for detectresourcemismatches
remove unused imports
initialize root cause of an exception
fix rebase with master
calculate checksum by all maven projects
set expiration policy for access tokens
remove a somewhat incorrect statement from the docs for conventionproperty
add setter and javadoc for packet reply timeout
unimplemented providers throw exception when used
add compatibility constructor
add summary of sstable version changes to descriptor
fix a couple of minor javadoc errors warnings
fix bad import
add reference to funnels from funnel javadocs
execute the wrapped runnable
fix clone to account for the update list add for update
changed the value to force saving
fix compile error
reverted the unintended removal of the call to container setsessiontransacted transacted
add todo for discussion of internal usage of identitysession
fix issue with remove of network connections though connection is active
only update wpcombloglist if the user has a valid dotcom token
enabled ignored test
fix salesforce security listener to replace oauth header
fix typo in servlet auto configuration javadoc
add some input checking to sequencer
smartlist set shouldn t increase modcount
fix inspection warnings
prefer druid curator compress to druid indexer runner compressznodes
fix proxy authentication in apacheasynchttpprovider
add isfavorite property to show
try extra getcolor
add missing return statement
fix compile error
fix tostring in payment channels for coin conversion
remove redundant modifier
missed npe check
set detached wallpaper on the scale up animation
bring snackbar to front after attaching to parent activity
made location clone public
asynccontext will throw exception
add ril error code list
smallfix remove explicit type argument in dataserverintegrationtest
assert statement formatting
do not set explicit preferred size for label
add failure message to test
replace loop with erasecolor
remove redudant close reason trim handled elsewhere now
add missing javadoc
set the listeners list back to null if all listeners are remove from it
llap zookeeper registry failures do not fail the daemon sergey shelukhin reviewed by prasanth jayachandran
remove the admin from the remove list when refreshing that admin
fix shard recovery tests
add some more pages
add type name
android update wxlistcomponent java
java comment the code to explain that the length of the strings need to be considered when encoding
have isindexbuilt expunge all tombstones
logging task class for better diagnostics
check more complete exception message for missing setting value
fix text and kovio links
remove intent logging from searchdialog java
made afterpropertiesset use instance variable instead of static variable
fix build broken documentation link
improve obsolete version exception message
make layoutanimationcontrollerassert final as the documentation suggests
updated ip address of localhost
fix typos in javadoc comments
fix property names for the asynchbase config in the tsdb ctor
don t call killtx if it has no work to do cleans up a misleading log line
handle empty arrival departure times
add hivetype tostring
fix missing strategy constants in json dump generator
make msessioncallback null in tvview unbindtvinput
add note to func
fix file comment larger timeout for a test
add super check
remove a duplicated statement that adds the menu items to the menu analyse
add more exception handling for invalid server lists
fix code checkstyle
hide tabs in execution ui if there are too many of them
making dto deserializable
suppress stack traces for known backup exceptions
send usage data for android
set the max size of a block in createfileoptions when creating a partition file in key value store
created getter setter for view field on remoteimageloaderhandler
fix wrong boundingbox transformation when using negative scaling values
hive schematool does not tolerate leading spaces in jdbc url chen xin yu via ashutosh chauhan
remove extra space
force the use of shm to avoid saturating disk i o
do not process properties themselves
bean component should use try conversion when choosing method based on parameter type matching
add set servletcontext method
fix javadocs and update copyright
fix race with loading last content fixes unfinished cacheupdater problem
remove the action link from the description of the problem
the failure message should have the response in a byte array
fixes an npe when compiling from the ide
fix imports in testcase
adds callbacks for settings
fix netty buffer leak
throw right exception
remove uneccessary call
fix lwjglgraphisc special case for idiotic mesa drivers that report the glx version before the opengl version
use the simplefuture parent pattern
fix update count fetch in statementresource
add static to some fields on blocktransformextenttest
remove assignment in psurfacejogl
fix npe in trimv4addrzeros
remove useless code
remove incorrect null check
prioritize user classpath
don t add things to the addons list if they don t have market links
add missing attribute copy
add equals and hashcode to windowed
throw exception when a bad adapter is passed to spinner
fix broken javadoc
don t log vcs exceptions unless asked to
fix a failing test
smallfix made field a local variable in abstractblockmetatest
add some basic print debugging support to fragment collections
add comments explaining that timesurfaces are now needed for travel time tiles
rev nik don t show packages for instance methods
fix issue on expand reported in ml
remove unused imports
ensure we use a mutable map to persist data
add synchronization statements in ostoragelocal addcluster to fix a concurrency problem on db open index rebuild
add comment on why some settings use the org hibernate prefix while others don t
add nullable notnull annotations
handle incorrect or missing style attribute
remove redundant initialization in retryhandlingblockmasterclient java
kill buildruleresolver buildandaddtoindex
make takeuntil obey rx contract
reset new boolean
add jedispool constructor that takes only host string
fix the keybinding array to the right one
fix race condition in testsqlstateexecution
fix quality flaw
add access to the jersey resourceconfig
fix defaultvalue regression
don t abuse erasure which should leave wildcard types as is
add reference for all jsonarray get and set methods
fix blank right panes on boot when old ui prefs are present
updated author tag
fix bug preventing selectionargs on rawquery from working properly
fix date interpreter in the sample app
add wheelmouse to reference
add the key to error message
add comment about why the instanceof is needed
add an operation to know if an index exists or not
commit correct document
temporarily skip simple cross origin progress events html for now
moved the graph related tests to their own class
avoiding use of return value of assignment operator
rename iodotnetty recycler maxcapacity default to iodotnetty recycler maxcapacity
fix windows slash issue with changedfile
remove dead import
when using external build do not validate configuration of old compilers
combine javadoc supplemented
fix multicirclejpacascadetest testpersistnocascadetotransient test
avoid npe if event don t has eventorigin annotation
fix a typo of fallbacktypeconverter
fix blank bubble after resume from unlock
add unit test
fix a small but nasty bug
making this internal method protected was public by mistake
fix bug on blob cluster import
add another test case for custom properties
add trace to exceptions
create graphfacadeimpl instance in build
prevent npe with session
remove todo marker for unimportant task of which it is also not
add jsp to supported file types
sets the interrupt status on interrupted exception if interrupted while waiting on channelfuture
fix a typo in settings documentation
fix a bug in searchingjavacompiler that was causing a npe
completely disable transactions for background threads
replayingdecoder should not throw an illegalstateexception even if a decoder implementation returns null when state transition occurred at least
actually handle results in onactivityresult
fix inserting wide char into column previously occupied by normal width char
fix minor typo
do not change table selection on drag n drop
add matrixtoimagewriter convenience class small code tweaks in javase make bytematrix width height property getters names standard
add license header
don t register the readlistener until after onopen has completed as messagehandler instances are likely to be registered during onopen
fix naming convention
change litebytestringutf8 to an error
remove white space
solved a fc problem caused by thread exits
fix fileserver example
throw exception in sendpacket if packet is null
moved system tables in ometadata
make oredictionary initvanillaentries private so that dumb modders will stop calling it
add colorint and drawableres to recommendations
fix a bug with how gcm service is constructed
don t leak lookup from recycled popups
fix npe in cylinderregion
add source active document with echo to dynamic commands
shorten the property name
changing stats getusersblogs call back to https
fix broken test flags in returned user object were not being copied from stored user
remove unnecessary ecs dependence
updated javadoc regarding previous deprecation of interestops value in channelinterestchanged event
invalidate parent when scrolling child view back onto screen
add thread safety test for constructorexpression
fix xml and json precondition
fix confussion with expected bsonobject
cdata brace highlighter in injected fragments
add pedestrian street to highway pedestrian reported by grant humphries
don t inline exported aliases
fix up a little test failure
remove unused var
allow types to customize the name displayed in describe
intermediate commit iterating on waitnotinloop
shadowshapedrawable extends shadowdrawable
deprecated dbcursor numgetmores and dbcursor getsizes
improve error message
log transaction code
only show open button as needed
a missed s byte bytebuffer substitution
skip null data in trim string
don t show lockscreen on signoutall and unencrypted
if moduletypemanager is not found like in upsource or some tests return empty type
remove superfluous limit
handling error for postgresql copy
add todo comment
add getviews view parent
disable fork mode if module was chosen successfully
better formatting for readpreference and writeconcern in tostring method
add missing bracket in javadoc
allow re enabling video track without transition through buffering state
remove unnecessary method
remove temp directories in tests
restored compatibility with kotlin plugin
fix window positioning code
include column size unit
fix jglfw minimize maximize snap on windows
fix pbos in downsamplesceneandupdateexposurenode
fix error when pac set rapidly
disable refresh sound when system disable ringer
remove unused non public method in db and unused private static field in dbcollection
make some public methods private
allow cleanup to echo rows again
do not set maxlifetime when disabling idle timeout
add a db style performance test
remove suppresswarnings that is no more needed
check port id for arc status update on hotplug
fix a bug with hide until toast
fix error message in cli
add fix to allow creation of models in saas signavio remove hypen from genereated uuid
temporary detached views not properly reported to accessibility layer
synchronized oncreate of shaderprogrammanager
remember src place order
add isbodyempty helper
expose unsafe copymemory
add more possible paths
start curator client
fix source formatting
fix multinomial gradient task reduce was add beta instead of gradient
don t show typos twice for injected code
removes redundant initialization within style builder
getprofiles should only returns non partial user info
fix javadoc typo
method parameter propogated for scala plugin
fix npe in localindexfragment in onpostexecute
add non processable method for testing
save use the filter selection in people page
fix escapejavascript tag add ajax events to other providers and corrected remotefields handling of parameters
fix classcastexception in sample app
add new generateexports option to api for closure compiler web services
ensure cursor is visible after execution point advances
add setitemstack itemstack stack to itemdrop
add local history actions to vcs popup even if vcs isn t configured
changed existing debug statement to info
remove flaky test
fix infinite recursive loop
update the comment which supposed to be checked in
relativelayout layoutparams should always have extended marginlayoutparams
fix two cases in the media grid where the swiperefreshlayout would display and never be dismissed
add annotations to required
improve the logs of json reports
rename method since equals has a specific meaning that doesn t seem the right fit for this use case
remove unused method
allow success examples to be empty null in docgen
retain iteration order when generating remove events for file snapshots that use absolute file paths
copy the scheduler properties to the consumer from the endpoint so that they get applied
fix reporting of window visibility in windowmanagerservice
remove bogus comment
log fix identifying root of node
revert fix freeze after selecting set wallpaper
using custom nodeidfactory is auto discovered with xml dsls
fix imports in testshadernodes
fix compilation issue
changed log tag to wordpress event
fix failing master
fix naming in queryresultcollection
unify test input locations
recompute slide offset from sliding view top when drag state becomes idle
add comment to test
handle unavailable port when not using dns to bootstrap resolve
fix dom implementation mapping
improved docs for anakia
fix sender bug
ignore connection termination test when using proxy
add validation for classdescriptor getdefaultobjectdescriptor new contract
allocate appropriate amount of memory in array concat
remove insertion action mode runnable on selection start
improve javadoc for handleradapter and handlermapping
fix block tests
fix remove debug output from jogl backend
revert improves list partition predicate
add httpd to log file types
update error message after template change
correct the comment
get headers with header parameter in lazy request
fix up mistake that broke all modloading
fix can t call back retry in expect the case of the call back method retry one by one
revert incorrectly applied ssr
fix license info
remove debug output
add test for no value cookies
fix wrong property filename in log by replace on constant
prevent name overflow
add null type resolution
remove unused imports
fix groovy version
fix problem with cast of storage
fix gpx casing
fix javadoc problems
don t allow manual sorting of featured lists
get messagesource by type instead of by name
add the constraint to avoid spaces in document s field names
make scrollbar widget aware of layout direction
xhrpolling infinity packet under heavy load bug fix
corrected layout offset for hidden tabs
also enable notifications by default in notificationsettings
tuning number of threads allocated in application pool
under certain server settings liferay may not clean up temp files when deploying portlets
fix unnecessary casts in servlet propertysource
throw an exception when an action is not found
deprecate preloadtexture make it a no op
fix issue with new stringtemplateparser
fix npe in wsoperation
addressing pr comment
remove unused imports
remove unnecessary check for retweeting protected account
add a newline to a javadoc sample in kafkaconsumer
fix issue with npe when generating saml attributes
fix network name not showing in qs sometimes
changed condition for spicemanager state machine
set background color as background for png with tranparency
layouts remove redundant code in generatedefaultlayoutparams
remove system out debug
fix cache hit calculation
add classloading logging when defaultnodeextension is loaded by multiple classloaders
offset base position when reading from slice
add stats for dataqueueproxy
smallfix remove public modifier from interface
add processinjectionbasedonservletcontext variant
don t consider the record functions to be structural
placed the completed call under the right braces
fix a compiler warning
updated stub to ensure compilability of python code
add very simple validation for missing github token
remove printstacktrace to fix cast exception
make sure we won t compile same modules in subsequent script runs in same vm instance
broadcaster add documentation
avoid division by zero when calculating huge zooms
reactivate devcockpit in pluginstest
do not change project name in eclipse
throwing appropriate converterexception
fix issue where onanimcycledone was not called when playing an animation
remove unused generics type from transportmessage
fix getrawstore method in abstractstoreclientfactory to bootstrap
specify all when quering history
add missing patch i didn t update from sources before merging
fix copy n paste error
fix cs issues
expose the server in serverstart event
allow exec types over the wire
fix checkstyle issues
fix the calculation of the cliptopamount which lead to holes
cosmetics remove unused imports
add missing fence
improved tests by closing all storages before the test
save off the seinfo value with packages list
we don t clone the first instance for cases where factories might not be set
add util method createstoragedirpath in tachyon util io fileutils java
rename ispristine to isimplicit
null context crashed getthumbnailforwpimagespan
deprecate duplicate token sets cleanup
simplify the sample
deleted some unuseful declarations
improved bounds checking and point nudging logic in gridsampler
improved compatibility of jobmanageritcase test with windows
resourcehttprequesthandler explicitly closes a resource s inputstream
remove unused field
fix damn import for the umpteenth time
fix modid field name for projectinjector
reuse resolve that does the same as the previous code
remove shameful system out println
add test in reflectionutil
ease defining breakpoint condition
fix license header
mark attrs nullable in toolbar constructors
it can contain any number of arguments depedning on the command edited
remove unused method
enforce valid range for priority
log transport connection pool configuration values
fix tiny issue with empty state graphic not being centered correctly
pre register the inner class as well
remove debug prints
meshbuilder add setuvrange
valueof should set word and value for the words it creates
fix npe in addcall qfix
show common part for trunk and branch with separate icon when highlight filter merge info
remove unnecessary this reference
make word completion work in custom file type files
add the sysroot argument even if it is the default for the particular crosstool
add a setmessageid method to mimemessage rather than just explicitly setting a header
use the correct import in tupledomainorcpredicate
cancel the timeout in any case
adds a check whether provider is registered in sip uri handler
remove unnecessary synchronized block
defend against null nodes in scope tree
move code to alert on the barrier inside the halt method on batchentryconsumer
handle exception on ins close
fix newchunk compression after recent fix
add missing headers for io classes
make comment consistent with the code
the sample should define unique ids
make preferred height at least as big as a line height
tidy up pairing mode configuration load save
fix typo for date time format specifiers
fix a memory leak in the media package report and fix thereof by michael koch
add missing copyright block
add missing line to license header
add a link to the java tutorial which inspired the qotm example
the member access checks must use the actual declaring class of the
restore instance state in correct order
fix crash when rapidly add remove sharedpreferences keys
add missing return statement
fix missing property
register new stuff in xmlcoreenvironment
fix unwanted default xml tag
fix another missing log
fix security check for private allocateappwidgetid api
add google internal memory consumption tests for immutable list set map bimap sortedmap sortedset
remove an unused import
removes signature not null precondition in sequencefileloader
project tree hanging fix
remove some leftovers in transactioncommitment
remote read no file
tidy up remove unused import from class extensioncompare
convert relative path to absolute path for local warehouse dir
simplify junit assertion
and add a new one
displaying the final file name
fix the javadoc for acquireexclusivelock
fix use of raw generics in minby maxby
reject update request that has both script and doc
fix sha1 for guava example
handle calendars with an access level of root
usagesstatistic must specify roaming disabled
starting graylog server in class rule
slightly update wording on graph equals javadoc
remove unused import
add exception euch
add mime type mapping json application json
stop load in imagepresenter if id or opener null
fix npe cause by race condition when axolotl service isn t initialized
add javadoc to test case
remove method specific javadoc
remove extraneous output
surely we just overlooked the proper generics on these signatures
assign back options correctly
remove system out
fix the cs error of camel components
client tests mock network internal address list leak fix
add stub of boolean class to environment
add beta disclaimer
improved doc in operation thread
remove eclipse generated comment
fix int division by zero when printing statistics
remove chronometer logspew
fix javadocs for deleteresponse isfound
remove progress printout
distinguish gridlayoutmanager from linearlayoutmanager
remove extra whitespace
fix build warning due to use of generics
throw better exception if allocation failed
you better not handle incomplete messages
make suppress duplicate a real diagnostic group
add user name of code
suppress noisy log line that is duplicated by localtransactionsigner
include buildconfig in the aar
fix transaction returning null
renamed getqconfig to getqueueconfig
fix type in exception
add principal in glassfishrequestupgradestrategy
remove dependency of zend ext
fix wrong import
fix bug with offset moving for iterating over repeating groups
delete unnecessary code correct qualified name will be passed to this place through presentablelookupvalue
handle interruptedexception in expressioninterpreter
implemented http configuration with http method
actually catch an oom in gettextfrompart
fix bug causing fusion table rows to get imported multiple times
add a helper method for path like env variables
occur warn reserve first same task warn last task
add the dstream join in java example
apply raymond borgan s fix for yed label associtations in graphml import
add proxy support
handle null value in csv output
add api to extend base panel in new project dialog
log full stack trace when there is an error may provide more clues as
create report files deeply mainy for testing purpises
add maxtransfers member to request class
add constants for unsigned max values
fix javadoc typos
add default label family in generating example
add attrib to pshape
add boolean support
remove needless static field for initialization in indexedringbuffer
fix log download bug where all nodes were getting the same zip file
make groovy shell action non dumb aware as it requires findclass
remove unused code
fix some bugs
fix maxidletime default to jetty s
add the default helpers as static objects as thy are quite useful to the outside world as it
log message if gcm token is empty
improve sample source
fix string concatenation with null literals
simplify client mode gossip handling
remove event bus call from the modelloader
fix npe in rabbitmq and polished the component
fix typo in palette doc
made toast property not final
adds incrbyfloat as cluster command
simplify map creation in testingconnectorsession
object delete in job
add various more browsers
improved getlabel error logging and fix warning in getstate error logging
don t flush index stamp cache too often
suppress unchecked warning for read method
codec memcache encode key directly no need to use unpooled
remove silly casts
fix issue with greater action when the xml tag has attribute with value
split the unreadable blog list strings into separate entries
add apanic kernel logging to checkinservice
prevent jline from expanding
fix failing test
fix some typos in mediatype javadocs and error messages
remove useless javadoc tags
use transparent scrollbars by default in simple editors without stripe markers too
use a better hash code for dfastate
when clear input content now show keyboard auto
remove mystery single quote
don t perform costly root model comparison when it s loaded the first time
add note that exception mappers can throw rather than return a
small tweak for the denon binding some devices do not support the device info url
methodparameter equals accepts subclasses pointing to same target declaration again
clarify variable name
remove error messages
the jaspic code now wants a servlet context which is relatively reasonable
add mpropersigningkeyset to packagekeysetdata constructor
mangling the logger name for better configurability
add protected httpclienttransportoverhttp2 gethttpclient accessor
throw a validation error in flow if any training data cols are non numeric
remove unnecessary use of reflection
fix db upgrade warnings
fix bug in remoteblockinstream
formatting input view on layout changes so as not to rely on child order
fix a byte size computation bug
when remove ldap settings also reset the ldapconnector
add a minimal null check to
change metadata caching policy to be daily by default
disable event expansion
remove error if created updated timestamp are missing it s not critical for the downloader to work
add issue number to the fixme marks
fix cs issue
ca catch patternsyntaxexception
fix broken column links in summary page
fix bug with layer selection
include file name for pdsc related error messages
make renamingmap serializable
i like touchy
use new okurlfactory in androidutils buildhttpurlconnection
fix code to get best relation token span
don t create anchors for fake cls stub based psi
add todo for missing suffixes jira follows
try invalidate in dashupdatesfragment
make webhistoryitem constructor public and hidden
add missing popup listener
describe non full hw level optional tag entries
set persist id on persisted input launch
improved error message in fuzzy k means
oops wrong variable
call finish instead of onbackpressed
suppressed expecting newline or semicolon error from js parser
remove accidental import
add access fields for second timezone columns
remove debug code from compilercontext
add some examples docs for the buildenvironment model
add missing type param
optimize wallpaper load avoid show black wallpaper
updated gremlin to play nice with the new titan
improve copystep s error message
fix failing test
clear the reference to the placeholder when not needed anymore
write virtual files in a write action
ignore constraint failures when populating search table
disable broken inspection by default
fix problem with objectfiletest on linux
remove unused part of code
fix bug in suffix patch
fix bug in bytearrayinput readlong
make pushbuilder s handling of http headers case insensitive
make getservername not return the port part of the host header
make order of modifiers consistent
remove unused node client setting
support libs remove setsupportschangeanimations method
tweaked loadclassforname separation of not found vs init errors
fix systemi info integration test
fix typo in maps immutableenummap
titled border should use current title
remove leading slash in resource path because classloader getresourceasstream should not have it
support the annotation suppresswarnings at class level
fix possible race condition
prevent misleading deployment group error
using different icon for the all episodes fragment
clarify nomenclature add phrasepenalty to baseline feature set
do not throw ioe if doing bindtoelement to a file
fix race at taskmanager registration during startup
fix recents row blinking when pip is focused
reparse injected files
add back support for properties type text channel mode in lexer actions
add getselectkeys to tablestatements
ignore faulty configuration
set the expiration time to the current time plus the expriation grace by default
add new image size columns for new install
leave the client udp connection connectionless and
add getclasssize to dexfilereader
fix compile error on eclipse
disable spellcheck in find in files dialog
fix exclude bug
fix broken storeview link on cluster status page
deferred types may delegate equals to their actual implementatinos
changed prestartupstoreupgrader from agpl to gpl license header to unbreak build
smallfix fix build error
improved assertion error message for a service registry create method returning null
remove unused import
plain text drop uses a different getter
provider does not tolerate null passwords well
fix possible npe
temporarily disabling a couple of tests that may have been fix by others to be re enabled after merging
missed a different mock signature
move some js modules to debug only
add rounded rect to papplet
change error message to an exception
remove classes that are no longer in the source tree
corrected logging message
propagate the original attributes
revert change from context to jmsendpoint an endpoint is sufficient and allows resue from activemq camel component
add javadoc for fsm conventions
add api access to reflectasm s methodaccess for event handling
better naming convention
remove szczepan s comment about explicit lock vs synchronised after discussion
remove unused field
can t use getfilestatus null to determine file existence use exists
fix dns a cname resolution bug
fix a potential double popup
fix a variable name reference in activity docs
remove old xbean references
remove deprecated call from headertransformer
fix import after okhttp update
add an awaitfix to indiceslifecyclelistenertests
use the getname method of principal rather than relying in tostring
fix tableinputformat returns no records when re opened with a new split
fix a constantpooltest failure
fix missing line return at eof
add a fake failure for testing ci
add final modifier to atomicboolean isstarted
fix npe when closing session that never authenticated
add the new tests into the suite
save splitter proportion in projectclosed not in dispose
fix benchmark app code
add lt to configure map menu
fix compilation error in test case and fix size of double typed constants
add new method dummy
changed size javadoc description and throw an illegalstateexception rather than a illegalargumentexception on getconstvalue
fix failing tests
fix cs error
fix private public
add missing javadoc for parameters
incorporated pr comments
add some incomplere javadoc for primitivevalue methods
fix scanner table name in treesync java so that it s looking at the meta data
share the daemon registry cache across daemon registry services instances to ensure serialised access to the cache storage
fix weird looking overlapping checkbox buttons
mongodb secured connections
change fileutils to be consistent with convention
skip shorter entities too
add rx utilities to reactivedialog
orc struct throws npe for tables with inner structs having null values prasanth j via gunther hagleitner
remove static hooking of clipboard manager again
remove unnecessary connection allocation
disable left click air task creation in sponge
super easy fix a style nit
add getters for success and failure handlers to abstractauthenticationprocessingfilter
remove dead load parm to getbootstraptokens
do expressions consistently with other engines
fix minor checkstyle warnings
prints stack trace for error which occurred during protocol sign in
reuse logic in overloaded methods
improved clientservice javadoc
include org gradle native dir as an immutable system property
remove auto reload of functions at every db open
wildcard query on non existent field matches all documents
maximize pane on chunk execution once per session
finish any enter animation when jumping to state
showing selfhosted if the email is not in wordpress com
adds a useful getkey implementation to generatedmethodbinding
don t use c style in java
add a non null default source pair for commontoken
fix things that should be static
handle nullpointerexception in cases where a proxy
remove unnecessary code
provide a default value for os name to avoid npe in some cases
fix silly copy paste mistake in lightnode
the jogl backend doesn t need null terminated strings for the attributes and the uniforms
jar file find assertion
use tabs option support
fix style builder parameter name
add getdisplaysize to uidisplayelement to remove lwjgl dependency
remove unnecessary code
remove debug log
javadocs fix javadoc examples in tumbling and sliding time windows
fix typo in string_cast operator
missed a line
made inner classes of partitiongenerateimpl static where possible
fix cs errors in camel http
remove unnecessary static qualifiers from enums
fix the test
only use hostname from srv lookup stick to user port setting
set buffer size to prevent chunked encoding
fix exception when recette palette is smaller than colors count
use other test port
haml lexer should generate pseudo start tag token for all tags not only implicit
insert manager ordering fix
fix kubernetes syntax
remove assertion since it could appear in normal situation
call refreshnotes in onresume so that the cursor is recreated
fix lbfgs multinomial lambda search do not use strong rules for now more predictors than columns in the dataset
add new layouttest fast dom deviceorientation basic operation html to skipped list
fix reference counting misaccounting
remove incorrect and unnecessary textconsolebuilder initialization
handle fs event queue overflow on windows fix
remove the istemplate value for android res values strings xml
remove useless imports
add missing channel to energy mod
add runwith annotation to concurrencytest
fix stack overflow
renamed oldprintstream to oldout to make the naming slightly
remove unused variable in test
fix bug in resize calculation
don t apply the same classinjector to a node twice
fix bug in outputemitter
add of a missing license header
remove unneeded cast
skip test in camel test blueprint that fails
fix illegalaccessexception in concurrentclosetest when run on robovm
fix qs expanding
add test for complete without close exception in abstractserverstream
add some documentation on serialization
fix compile issue with desktopgeneratorcontroller
test use nomergepolicy rather than nomergescheduler in freqtermsenumtests
fix graph repair timing
use parameterized logging
make hystrixcommandproperties backwards compatible for classes which extend it
fix scrolling in the poi editor
add debugging information when a session was not found
add checkreturnvalue to binarypredicates and defaults
improve javadoc of http connection
enable debug logging in blinking test
check isbreakchar in getwrappedbounds
fix multiple manage catalogs items issue
add missing assert in calloptionstest
add the hasgravity method to blocks
check for null code item
rename not_found_marker and make it private
binary stub decompiler get project
gui show an error message if we can t set the look feel
allow dialect selection
catch uncaught ioexception
add permission for worldgeneratorpluginlibrary
remove the sytem out line of aggregationstrategywithpreservationtest
replace any number of millisecond characters with at most three of them
add test of reused ssl socket factory
temporarily increase the number of connections in the test
fix a regression in view cancellongpress
add some comments
add pre destroy
replace notimplementedexception with unsupportedoperationexception
remove deprecated and duplicated class acceptencodinggzipinterceptor
remove unused code
bleeding add inventory
fix up more conjunct adjective phrases from ancora multiwords
use the atomic integer counter to represent the size
fix problem with seeking before starting playback
fix vcard so that it looks at im correctly
avoid npe when sending buffered data from the server there is no mask in this case
fix an index out of bounds exception
fix up concurrent modification exception ticks for players now occur on both sides
trim boolean configuration before parsing
fix flow shutdown bug
allow tabs and newlines to count as whitespace
remove unnessary imports
print right side of range in nodeprobe ring output
add patch to prevent the indexoutofbounds in stackblurmanager
avoid null node names in obj importer
revert improves index lookup performance by one order of magnitude
fredricho jonathanl fix issue with views not recycling on media grid
java double the length of max event length
make sure testephemeralfilechannel listfiles uses correct root file path prefix on windows
fix preferred focus
adjust header names
don t include empty fingerprints in list of otrfingerprints
fix bug on propagating commands among the cluster nodes
tolerate amazonclientexception not just amazonserviceexception
performance dispose light nodes in hotspots
update java docs
only set results if views available in addfragments
don t show reengagement reminders if reminders are disabled
updated license header
do not add subpackages that are not found
remove checkstyle suppression
retrieve true underlaying class properly handling cglib enhanced versions
update action presentations even if there are no mouse or key events
provide hint to use stacktrace to see the full stack trace
cleaned up comment
listcollectionsoperation execute now uses the right connection to execute the listconnections command
skip requestrectangleonscreen for webtextview during
redissonlist contains method speed optimization
delete file view api leftovers
add commits prefix to pager label
use default constructor to no tie it to a specific version
add a little comment about the enabled value
adds npe check
use the standard rejectedexecutionexception
generify a little bit
don t offer initiator his own candidates
fix touch handling in the zoomabledraweeview
fix checkstyle errors
corrected a typo in comment
fix constructor name
matching degree in nameutil
fix some minor javadoc errors in accountmanager
remove unused context provide method
fix infinite loop when receiving a settings frame
fix flag_privileged numbering
prevent rare disastrous classloading in first call to locksupport park
avoid npe on null group metadata
add final zero padding test
corrected config comment on dochunkloading
remove empty line
updated configuration keys to new naming
working on macosx java resize bug
add api to access loadbalancerexecutor from loadbalancingrxclient
revert add disconnect method
don t show module suggestions if tests were interrupted e g stopped before tests were collected
fix js backend broken recently
smallfix remove explicit argument type in fileinstreamintegrationtest
open links in new browser
fix connection leak thanks tuhu
metadata store is not a redirecting store
add log to all kind of loadmetadata
allowing basebitmaptextureatlassourcedecorator ondecoratebitmap to throw exception
updated the description for the rewrite option
fix annotation style
migrate user guide from docbook to asciidoctor findbugs fix
git fix javadoc
add link to view pager guide
add missing withicon method with an drawableres
fix import order warning from checkstyle
beeline always return the same error message twice chaoyu tang via ashutosh chauhan
add better support for launchers that don t resize on initial placement
provide ability to specify name based rules
daemonizing the sts thread used for the credential refresh
revert add milliseconds per second constant
fix edge condition when checking number of pages
correct an issue with outdated prev_application_mode after changes of app profile via quickchanger and subsequent only show routing
remove ril_request_set_radio_mode and ril_request_data_idle commands
fix npe in error logging thanks mihail
remove unnecessary import
do not merge only throw exception in ebt helper if parent bounds are untouched
revert map marker color change
made state in indexstatisticstest thread safe
git remove unnecessary method overriding
validate function arguments
cherry pick load the webview java code from an apk
ignore flaky defaultserverfactorytest testgracefulshutdown
support forced resolve for lazy flexible types
fix visibilitymodifier allowpublicimmutablefields work in enums
add support for applicationinfo to shadowresolveinfo
add a couple of missing nullable annotations
git wrap the vcsexception not to loose the cause
not wrap _hintfields into unmodifiable to prevent npe
extend zoom level for cykloatlas cz
fix quality flaw
suspend animator ticks while not necessary
remove java awt as a default import in the standardtypelocator
add management center config
cleaned up log statement
moved impsaddressutils to
add missing apache headers
add workinf folder creation to test
handle bug with byte in extras
correctly retrieve dom children for elements with no tag
log directory creation failure not success
avoid shared lexer usage from different threads
cast range to number instead of double
fix issue tracking regression after last optimizations
fix back arrow on home
remove work around for driver state change
fix pointer index out of range exception do not merge
fix pmd warning if statement without curly braces
updated test to run even when not connected to the network
fix early session termination in falsingmanager
fix min max calculation for dynamic graphs in ranking
add unit for finding parent resource
remove printline add more explicit throwable unboxing in task exceptionally handler
rename replacement for the node gettype method to node gettoken
add correct scope annotation
ignore balnace class related parameters for dl ae or all unsupervised models built by supervisedmodelbuilder
fix testing metadatautil for varchar
changed logcat to verbose and add time
throw filecomparisonfailure instaed of comparisonfailure
fix broken refactor
remove null check for server response since null apparently is a valid response
fix up issue with jsp taglibs
java only do a single volatile lookup
remove confusing comment in paymentgatewayprocessor
remove useless try catch
fix up privacy derp in tickevent
add move test
rolevoter can improve performance
add isclosed method to asyncbatchcursor
improve doc for get set hintavoidbackgroundclipping
fix compilation error
fix error message in case fetcher raises an exception
add roster versioning to calculation of server compat
add implementation of sendstickybroadcast in shadowappplication class
add missing comment
changed lib path
update blockjunit4classrunner to no longer use the deprecated failontimeout ctor
remove unnecessary space from debug message
fix gl version check in joglgraphics
wildfire would have jiveversion not ofversion
fix variable scope
printed a warning message when timing tests are disabled
add assertions to test
improved handling of delegate paths
fix copy paste error thanks kkolinko
carriertext clear itself when no mob
declaring a variable always returns a value now
updated javadoc comments on flash scope
add todo for bip
don t zap more page memory to zero than is strictly necessary
set the messageid header rather than add it when restoring from the local store
move load fence to public method as not need on private usage
azure revert discovery plugins must declare their unicasthostsprovider
fix a crash
remove unused imports
allow manual sync even when background data is disabled
add varargs requestparams convenience constructor
make system properties override properties file
add an internetdomainname test that confirms proper behavior when under tld patterns overlap
rename actions back to admin and add suffix s instead
add an extra intent parameter for recording duration limit
android update version
ignore broken tools each time to be able to createtools concurrently
improved javadoc transactionaltask
hide actions of episode details fragment
disable disk threshold allocation decider for crate test cluster
changed kdtree s xyzpoint fields into protected
relax whole word criterion
improve prototype javadoc
clarify enablescheduling javadoc
fix contentview parent
delete a log line in ssowebviewclient
disable imageio s file based cache should speed up texturepack loading stitching
fix progressbar logic for animationdrawables
add current aosp class name for opensslsocketimpl
fix issue in average pooling backprop
move log buffer status field onto its own cacheline
save xml after setting names
fix commented code
fix nullpointer in liveusers
allow any whitespaces as separator
working integration tests
check if the plugincomponent is null before proceeding and firing events
quickfix made recursive enumeration parameter flag public in fileinputformat
fix bug in equals from prior change
fix crash for showing linker errors for apps with no label
prevent npe in glm pojo creation
override nfeatures method in dlmodeloutput
add aux field to localactor used by pulsar
confusing bold projectname string for projects linked from outside r akirillin
remove unused api in ff to cnn processor
close the inputstream that was used to get the content of
fix possible npe
remove test on windows
adapting error message
remove unused variant of resolvecontext resolve static method
add better index usage code
use existing project
do not cancel import task during auto import
throw facebookexception if attributionidentifiers is null when it is needed and protect against null if it is optional
managedperformancecountermbean should extend managedcountermbean
allow null to be re bound as a root script
prevent editor load issues associated with yaml parsing
don t append comma in case of no shortcuts
fix a test
avoiding oom due to scheduled update of completion progress
add lock domain and magic create domain to known hosts
using correct section
improved object namel formatting
disable recoverywhileunderloadit for now
add missing javadoc
allow esc key to interrupt the console even if used in the source pane
do not update viewport border for custom border
don t track offsets after completion when psi has become invalid
fix sometimes dead sleep after seekto
make the silent vibrate status icon reflect overall device state
only create bouncycastleprovider the first time
remove stray after splitting of increment and decrement methods
type may be weakened readobject writeobject
fix bars jumping to black on activity launch
setting mbackdropback to null if artwork isn t available
fix input javadocs
add emma filter settings
create should respect overwrite
used request extractor interface for content extractor
the progressstart now also reset the current offset
add support for new wallpaper theme attribute
fix pystubstest testrenamingupdatesthestub
do not merge tweak overscroll
fix cut paste stupid mistake
remove early exit to not update progressindicator
remove warning by add serialversionuid
apply the change from upstream master by add more comments on security on master
fix move to top bottom
add javadoc for peek_height_auto
avoided increased memory consumption
add missing showall for caught exception
modified comment that said the iq packets couldn t have extensions
read madvoc annotation result
avoid induced npe
don t use system module
add lookup key to callerinfo
add some comments
remove nonexistent global variable from test
remove unused imports
disabling the exchange cache it was causing the concurrent tests to fail
fix npe if from to is missing
reversing immediately after starting animator ends it
push don t nullize border in renderer
fix avoid icon for context menu
fix properties in tostring for testingconnectorsession
add setter for server parameter
improve soft wraps performance
fix tablename test
add date to logentry
trim html headers from template remove end and treeend directives
fix problem with default block size in hdfs binding
define acl acronym on first usage
fix a typo in flickr downloader
fix method access
fix javadoc of org sonar api server ws webservice
add test for stringutils replace
add deserializers to class loader resolver
add psk as option for second element in handshake algorithm pattern
corrected views visibility on hide bubbles
fix bug on inheritedrole
fix usage of deprecated jackson yaml code
ignore graphs that don t support ids
add double tap at top or bottom to jump to the top or bottom of the message
delete unused imports in headerassertiontests
cannot delete a library not used in the project
fix after project close exception
fix javadoc of sethandlemethods
fix problem waking screen from negative proximity sensor events
remove some dead code
fix compilation issue was still a reference to the remotepropagatable
file history reorder statements
fix crash in setadditionalinputmethodsubtypes
get progress information in sql interpreter
fix due to lombok change in plainstreetedge
replaced deprecated gridmarkerclusterer with radiusmarkerclusterer
add layout xml and strings xml to important files node in android projects
bring documentation of service
remove extra from match original
